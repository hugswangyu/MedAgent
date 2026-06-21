"""离线预处理：将疾病节点的长文本属性拆分为 Chunk 节点并建向量索引。

处理属性：疾病病因、预防措施、疾病简介（属性值 > 200 字才拆分）
Chunk 节点标签：KGChunk
关系：(疾病)-[:HAS_CHUNK {attribute: '疾病病因'}]->(KGChunk)
向量索引：kg_chunk_vector（Neo4j 5.x 原生，cosine，dim=512）

每个长属性产出两类 KGChunk：
  - level='detail'  ：细分块（order>=0），供"细化型" query 余弦召回；
  - level='summary' ：提纲摘要（order=-1），供"概括型" query 返回，
                       长度恒定在 ~400 字，与原文长度无关，避免撑爆上下文。

用法：
    uv run python scripts/build_kg_chunks.py
    uv run python scripts/build_kg_chunks.py --dry-run   # 只打印统计，不写库
    uv run python scripts/build_kg_chunks.py --reset     # 清空已有 KGChunk 重新跑
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import py2neo
from tqdm import tqdm

from medrag.config.settings import settings
from medrag.llm import get_llm_client
from medrag.prompts import KG_SUMMARY_PROMPT_TEMPLATE
from medrag.vectors.embedding import EmbeddingModel

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
LONG_ATTRS = ["疾病病因", "预防措施", "疾病简介"]
THRESHOLD = 200        # 超过此字数才拆分
MAX_CHUNK = 300        # 单块最大字数
MIN_CHUNK = 50         # 单块最小字数（过短则合并）
OVERLAP = 30           # 相邻块首尾重叠字数
EMBED_BATCH = 64       # 每批 embedding 的 chunk 数
WRITE_BATCH = 50       # 每批写入 Neo4j 的节点数（embedding 数组较大，批次不宜太多）
DISEASE_BATCH = 100    # 每批处理的疾病数：每批算完即落库，断点续跑的提交粒度
EMBEDDING_DIM = 512    # BGE-small-zh-v1.5

# 语义边界正则：换行后紧跟编号/标题
_SEMANTIC_SEP = re.compile(
    r'\n(?='
    r'[一二三四五六七八九十百]{1,3}[、．.]'   # 一、二、
    r'|[（(][一二三四五六七八九十百\d]{1,3}[）)]'  # （一）（1）
    r'|\d{1,2}[、．.]'                        # 1、2.
    r'|第[一二三四五六七八九十百\d]{1,3}[章节步]'  # 第一节
    r')'
)

# ---------------------------------------------------------------------------
# 切块逻辑
# ---------------------------------------------------------------------------

def _split_by_sentence(text: str, max_len: int, overlap: int) -> List[str]:
    """按句号切，超长段的兜底方案。"""
    chunks: List[str] = []
    buf = ""
    for sent in re.split(r'(?<=[。！？；])', text):
        if not sent.strip():
            continue
        if len(buf) + len(sent) <= max_len:
            buf += sent
        else:
            if buf:
                chunks.append(buf.strip())
            # 重叠：把上一块末尾 overlap 字带入下一块
            tail = buf[-overlap:] if len(buf) > overlap else buf
            buf = tail + sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _merge_small(chunks: List[str], min_len: int) -> List[str]:
    """把过短的块合并到下一块。"""
    merged: List[str] = []
    pending = ""
    for c in chunks:
        pending = (pending + c).strip() if pending else c
        if len(pending) >= min_len:
            merged.append(pending)
            pending = ""
    if pending:
        if merged:
            merged[-1] = (merged[-1] + pending).strip()
        else:
            merged.append(pending)
    return merged


def chunk_text(text: str) -> List[str]:
    """主切块函数：语义边界优先，字数兜底。"""
    if not text or len(text) <= THRESHOLD:
        return [text.strip()] if text and text.strip() else []

    # 第一层：按语义编号边界切
    raw_segs = _SEMANTIC_SEP.split(text)
    if len(raw_segs) == 1:
        # 无语义边界，按 \n 段落切
        raw_segs = [s for s in text.split('\n') if s.strip()]

    # 第二层：超长段再按句号切
    fine: List[str] = []
    for seg in raw_segs:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) <= MAX_CHUNK:
            fine.append(seg)
        else:
            fine.extend(_split_by_sentence(seg, MAX_CHUNK, OVERLAP))

    # 第三层：合并过短块
    return _merge_small(fine, MIN_CHUNK)


SUMMARY_MAX = 600      # 摘要最大字数（概括型 query 返回它，长度受控）
_SECTION_SNIPPET = 80  # 提纲法每个小节保留的字数（标题 + 开头说明）


def build_outline(text: str) -> str:
    """提纲式摘要（无 LLM 兜底）：概括型 query 返回它，避免塞入万字原文。

    优先利用「一、二、（一）1.」等编号小节：每节取开头一小段（标题+起始说明，
    而非仅标题，避免过薄）；无结构时退化为按句号累积前若干句。
    结果长度受控在 ~SUMMARY_MAX 字，与原文长度无关。
    """
    if not text or not text.strip():
        return ""

    segs = _SEMANTIC_SEP.split(text)
    if len(segs) > 1:
        # 有编号结构：每节取开头 _SECTION_SNIPPET 字（含标题与起始说明）
        parts: List[str] = []
        used = 0
        for seg in segs:
            seg = re.sub(r"\s+", "", seg.strip())
            if not seg:
                continue
            snippet = seg[:_SECTION_SNIPPET]
            parts.append(snippet)
            used += len(snippet)
            if used >= SUMMARY_MAX:
                break
        outline = "；".join(parts)
        if outline:
            return outline[:SUMMARY_MAX]

    # 无编号结构：按句号累积到 SUMMARY_MAX
    acc = ""
    for sent in re.split(r"(?<=[。！？])", text):
        if not sent.strip():
            continue
        if len(acc) + len(sent) > SUMMARY_MAX:
            break
        acc += sent
    return (acc or text[:SUMMARY_MAX]).strip()


# 关闭「思考模式」的额外参数（按模型，经 DashScope）：thinking 模型开思考会慢
# 20 倍（glm-5 实测 119s → 5s），摘要任务不需要思考。不同厂商参数名不同。
_NO_THINK_EXTRA = {
    "glm-5": {"thinking": {"type": "disabled"}},
    "glm-5.1": {"thinking": {"type": "disabled"}},
    "qwen3.6-flash": {"enable_thinking": False},
    "qwen3.5-flash": {"enable_thinking": False},
    "qwen3.6-27b": {"enable_thinking": False},
    # deepseek-v3 / v4-pro / v4-flash / kimi 默认即非思考，无需额外参数
}


def _clean_summary(s: str) -> str:
    """清洗 LLM 输出里的 markdown/前缀（如 kimi-k2.6 爱加 **加粗**）。"""
    s = s.strip().strip("`").strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)       # **加粗**
    s = re.sub(r"(?m)^#{1,6}\s*", "", s)          # # 标题
    s = re.sub(r"^(概览|摘要|总结)[：:]\s*", "", s)  # 残留前缀
    return s.replace("**", "").strip()


def _llm_summarize(text: str, disease: str, attribute: str) -> str | None:
    """用 LLM 把长属性原文概括成一段概览，长度按复杂度自适应。

    按 ``settings.kg_summary_models`` 依次尝试（均经 Qwen/DashScope 调用，
    仅模型名不同），前一个欠费/失败自动降级到下一个；全部失败返回 None，
    由调用方退回提纲法。思考型模型统一关闭思考以提速。
    """
    if len(text) > settings.kg_summary_complex_threshold:
        target = ("用约 400–600 字，分条覆盖各主要方面，并对每一方面用一句话"
                  "说明其要点或机理（如某因素如何导致该病）")
    else:
        target = "用约 200–300 字精炼概述要点"

    prompt = KG_SUMMARY_PROMPT_TEMPLATE.format(
        disease=disease, attribute=attribute, target=target, text=text,
    )

    client = get_llm_client("qwen")
    for model in settings.kg_summary_models:
        model = model.strip()
        if not model:
            continue
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        extra = _NO_THINK_EXTRA.get(model)
        if extra:
            kwargs["extra_body"] = extra
        try:
            resp = client.chat.completions.create(**kwargs)
            out = _clean_summary(resp.choices[0].message.content or "")
            if out:
                return out
        except Exception as exc:  # 欠费 / 限流 / 网络 → 降级到下一个模型
            tqdm.write(f"  [摘要降级] {model} 失败（{exc}），尝试下一个")
            continue
    return None


def make_summary(text: str, disease: str, attribute: str, use_llm: bool) -> str:
    """生成概览。

    - 原文 ≤ ``kg_summary_llm_min``：本身够短，直接拿原文作概览，不调 LLM；
    - 原文更长：LLM 浓缩为主、提纲法兜底。
    """
    if len(text) <= settings.kg_summary_llm_min:
        return text.strip()
    if use_llm:
        s = _llm_summarize(text, disease, attribute)
        if s:
            return s
    return build_outline(text)


# ---------------------------------------------------------------------------
# Neo4j 操作
# ---------------------------------------------------------------------------

def _connect() -> py2neo.Graph:
    return py2neo.Graph(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        name=settings.neo4j_database,
    )


def _fetch_diseases(g: py2neo.Graph, limit: int = 0) -> list[dict]:
    cypher = "MATCH (d:疾病) RETURN d.名称 AS name, " + ", ".join(
        f"d.`{a}` AS `{a}`" for a in LONG_ATTRS
    )
    if limit and limit > 0:
        cypher += f" LIMIT {int(limit)}"
    return g.run(cypher).data()


def _already_chunked_names(g: py2neo.Graph) -> set[str]:
    rows = g.run("MATCH (c:KGChunk) RETURN DISTINCT c.disease_name AS n").data()
    return {r["n"] for r in rows if r["n"]}


def _clear_chunks(g: py2neo.Graph) -> None:
    g.run("MATCH (c:KGChunk) DETACH DELETE c")
    print("已清空所有 KGChunk 节点")


def _write_batch(g: py2neo.Graph, records: list[dict]) -> None:
    """批量写入 KGChunk 节点并建立 HAS_CHUNK 关系。"""
    g.run(
        """
        UNWIND $rows AS row
        MATCH (d:疾病 {名称: row.disease_name})
        MERGE (c:KGChunk {chunk_id: row.chunk_id})
        SET c.disease_name = row.disease_name,
            c.attribute    = row.attribute,
            c.text         = row.text,
            c.order        = row.order,
            c.level        = row.level,
            c.embedding    = row.embedding
        MERGE (d)-[:HAS_CHUNK {attribute: row.attribute}]->(c)
        """,
        rows=records,
    )


def _create_vector_index(g: py2neo.Graph) -> None:
    g.run(
        f"""
        CREATE VECTOR INDEX kg_chunk_vector IF NOT EXISTS
        FOR (c:KGChunk) ON (c.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
    )
    # 等待索引就绪
    for _ in range(30):
        rows = g.run(
            "SHOW INDEXES WHERE name = 'kg_chunk_vector'"
        ).data()
        if rows and rows[0].get("state") == "ONLINE":
            return
        time.sleep(2)
    print("警告：索引创建超时，请手动检查 Neo4j 索引状态")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写库（摘要走提纲法预览，不调 LLM）")
    parser.add_argument("--reset", action="store_true", help="清空 KGChunk 后重跑")
    parser.add_argument("--no-llm", action="store_true", help="摘要只用提纲法，不调 LLM")
    parser.add_argument("--workers", type=int, default=0,
                        help="摘要并行线程数（默认取 settings.kg_summary_workers）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 个疾病节点，用于小样本验证（0=全量）")
    args = parser.parse_args()

    # 概览摘要：正式跑用 LLM（提纲法兜底）；dry-run / --no-llm 仅用提纲法
    use_llm = not args.dry_run and not args.no_llm
    if use_llm:
        print(f"概览摘要使用 LLM 链：{' → '.join(settings.kg_summary_models)}"
              f"（经 Qwen/DashScope，失败自动降级）")
    else:
        print("概览摘要使用提纲法（不调 LLM）")

    print("连接 Neo4j…")
    g = _connect()

    if args.reset:
        _clear_chunks(g)

    print("加载疾病节点…")
    diseases = _fetch_diseases(g, limit=args.limit)
    if args.limit:
        print(f"小样本模式：仅处理前 {len(diseases)} 个疾病节点")
    print(f"共 {len(diseases)} 个疾病节点")

    skip_names = set() if args.reset else _already_chunked_names(g)
    if skip_names:
        print(f"跳过已处理的 {len(skip_names)} 个疾病（使用 --reset 重跑全量）")

    print("加载 embedding 模型…")
    embedder = EmbeddingModel()

    llm_min = settings.kg_summary_llm_min
    workers = max(1, args.workers if args.workers else settings.kg_summary_workers)

    def _summary_record(name_: str, attr_: str, text: str) -> dict:
        return {
            "chunk_id": f"{name_}__{attr_}__summary",
            "disease_name": name_,
            "attribute": attr_,
            "text": text,
            "order": -1,
            "level": "summary",
            "embedding": None,
        }

    def _gen_heavy(task: tuple[str, str, str]) -> dict | None:
        name_, attr_, val_ = task
        text = make_summary(val_, name_, attr_, use_llm)
        return _summary_record(name_, attr_, text) if text else None

    def build_pending(batch: list[dict]) -> list[dict]:
        """为一批疾病构造全部待写节点（detail 块 + summary）。"""
        pending: list[dict] = []
        summary_tasks: list[tuple[str, str, str]] = []
        for disease in batch:
            name = disease["name"]
            for attr in LONG_ATTRS:
                val = disease.get(attr) or ""
                if len(val) <= THRESHOLD:
                    continue
                # detail：细分块，供细化型 query 余弦召回
                for i, text in enumerate(chunk_text(val)):
                    pending.append({
                        "chunk_id": f"{name}__{attr}__{i}",
                        "disease_name": name,
                        "attribute": attr,
                        "text": text,
                        "order": i,
                        "level": "detail",
                        "embedding": None,
                    })
                summary_tasks.append((name, attr, val))

        # 短属性原文直接作概览（零成本）；长属性走 LLM（提纲兜底）
        plain = [t for t in summary_tasks if len(t[2]) <= llm_min]
        heavy = [t for t in summary_tasks if len(t[2]) > llm_min]
        for name_, attr_, val_ in plain:
            if val_.strip():
                pending.append(_summary_record(name_, attr_, val_.strip()))
        if heavy:
            if use_llm and workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    recs = list(ex.map(_gen_heavy, heavy))
            else:
                recs = [_gen_heavy(t) for t in heavy]
            pending.extend(r for r in recs if r)
        return pending

    def embed_and_write(pending: list[dict]) -> int:
        """对一批 pending 节点做 embedding 并写入 Neo4j（写完即持久化）。"""
        if not pending:
            return 0
        texts = [p["text"] for p in pending]
        embs: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            embs.extend(embedder.encode(texts[i: i + EMBED_BATCH], is_query=False))
        for p, e in zip(pending, embs):
            p["embedding"] = e
        for i in range(0, len(pending), WRITE_BATCH):
            _write_batch(g, pending[i: i + WRITE_BATCH])
        return len(pending)

    # 只保留未处理的疾病（断点续跑：skip_names 来自库中已有 disease_name）
    todo = [d for d in diseases if d["name"] and d["name"] not in skip_names]
    print(f"待处理 {len(todo)} 个疾病（跳过已完成 {len(diseases) - len(todo)} 个）")

    # ---------- dry-run：只取前若干疾病构造样例，不写库 ----------
    if args.dry_run:
        sample = build_pending(todo[:50])
        for attr in LONG_ATTRS:
            items = [p for p in sample if p["attribute"] == attr]
            n_sum = sum(1 for p in items if p["level"] == "summary")
            n_det = sum(1 for p in items if p["level"] == "detail")
            print(f"\n【{attr}】summary {n_sum} 条 / detail {n_det} 块：")
            for s in [p for p in items if p["level"] == "summary"][:1]:
                print(f"  ▶ summary [{s['disease_name']}] ({len(s['text'])}字): {s['text'][:120]}…")
            for s in [p for p in items if p["level"] == "detail"][:2]:
                print(f"  · detail [{s['disease_name']}] #{s['order']} ({len(s['text'])}字): {s['text'][:60]}…")
        return

    if not todo:
        print("无待处理疾病，确保索引后退出")
        _create_vector_index(g)
        return

    # ---------- 分批：每批 切块+摘要+embed+写库 一次性提交（写完即持久化）----------
    print(f"\n开始分批建库（每批 {DISEASE_BATCH} 个疾病，写完即落库，可断点续跑）…")
    total_written = 0
    batches = [todo[i: i + DISEASE_BATCH] for i in range(0, len(todo), DISEASE_BATCH)]
    for batch in tqdm(batches, desc="建库(疾病批)"):
        pending = build_pending(batch)
        total_written += embed_and_write(pending)
        tqdm.write(f"  已落库 {total_written} 个节点")

    # ---------- 建向量索引 + 验证 ----------
    print("\n创建向量索引 kg_chunk_vector…")
    _create_vector_index(g)
    cnt = g.run("MATCH (c:KGChunk) RETURN count(c) AS n").data()[0]["n"]
    idx = g.run("SHOW INDEXES WHERE name = 'kg_chunk_vector'").data()
    idx_state = idx[0]["state"] if idx else "未找到"
    print(f"\n完成！本次写入 {total_written} 个节点")
    print(f"  KGChunk 节点总数：{cnt}")
    print(f"  向量索引状态：{idx_state}")


if __name__ == "__main__":
    main()
