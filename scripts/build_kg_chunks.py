"""离线预处理：将疾病节点的长文本属性拆分为 Chunk 节点并建向量索引。

处理属性：疾病病因、预防措施、疾病简介（属性值 > 200 字才拆分）
Chunk 节点标签：KGChunk
关系：(疾病)-[:HAS_CHUNK {attribute: '疾病病因'}]->(KGChunk)
向量索引：kg_chunk_vector（Neo4j 5.x 原生，cosine，dim=512）

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
from pathlib import Path
from typing import List

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import py2neo
from tqdm import tqdm

from medrag.config.settings import settings
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
WRITE_BATCH = 200      # 每批写入 Neo4j 的节点数
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


def _fetch_diseases(g: py2neo.Graph) -> list[dict]:
    cypher = "MATCH (d:疾病) RETURN d.名称 AS name, " + ", ".join(
        f"d.`{a}` AS `{a}`" for a in LONG_ATTRS
    )
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
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写库")
    parser.add_argument("--reset", action="store_true", help="清空 KGChunk 后重跑")
    args = parser.parse_args()

    print("连接 Neo4j…")
    g = _connect()

    if args.reset:
        _clear_chunks(g)

    print("加载疾病节点…")
    diseases = _fetch_diseases(g)
    print(f"共 {len(diseases)} 个疾病节点")

    skip_names = set() if args.reset else _already_chunked_names(g)
    if skip_names:
        print(f"跳过已处理的 {len(skip_names)} 个疾病（使用 --reset 重跑全量）")

    print("加载 embedding 模型…")
    embedder = EmbeddingModel()

    # ---------- 收集所有需要写入的 chunk ----------
    pending: list[dict] = []   # 待 embed + 写入
    total_chunks = 0
    skipped_diseases = 0

    for disease in tqdm(diseases, desc="切块"):
        name = disease["name"]
        if not name or name in skip_names:
            skipped_diseases += 1
            continue

        for attr in LONG_ATTRS:
            val = disease.get(attr) or ""
            if len(val) <= THRESHOLD:
                continue
            chunks = chunk_text(val)
            for i, text in enumerate(chunks):
                pending.append({
                    "chunk_id": f"{name}__{attr}__{i}",
                    "disease_name": name,
                    "attribute": attr,
                    "text": text,
                    "order": i,
                    "embedding": None,   # 占位，后面填
                })
            total_chunks += len(chunks)

    print(f"\n切块完成：{total_chunks} 个 chunk（跳过 {skipped_diseases} 个已处理疾病）")

    if args.dry_run:
        # 打印样例
        for attr in LONG_ATTRS:
            samples = [p for p in pending if p["attribute"] == attr][:2]
            print(f"\n【{attr}】样例（共 {sum(1 for p in pending if p['attribute']==attr)} chunks）：")
            for s in samples:
                print(f"  [{s['disease_name']}] chunk#{s['order']} ({len(s['text'])}字): {s['text'][:80]}…")
        return

    if not pending:
        print("无需处理，退出")
        return

    # ---------- 分批 embed ----------
    print(f"\n开始 embedding（batch={EMBED_BATCH}）…")
    texts = [p["text"] for p in pending]
    all_embeddings: list[list[float]] = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="embedding"):
        batch = texts[i: i + EMBED_BATCH]
        all_embeddings.extend(embedder.encode(batch, is_query=False))

    for p, emb in zip(pending, all_embeddings):
        p["embedding"] = emb

    # ---------- 分批写入 Neo4j ----------
    print(f"\n写入 Neo4j（batch={WRITE_BATCH}）…")
    for i in tqdm(range(0, len(pending), WRITE_BATCH), desc="写入"):
        _write_batch(g, pending[i: i + WRITE_BATCH])

    # ---------- 建向量索引 ----------
    print("\n创建向量索引 kg_chunk_vector…")
    _create_vector_index(g)

    # ---------- 验证 ----------
    cnt = g.run("MATCH (c:KGChunk) RETURN count(c) AS n").data()[0]["n"]
    idx = g.run("SHOW INDEXES WHERE name = 'kg_chunk_vector'").data()
    idx_state = idx[0]["state"] if idx else "未找到"
    print(f"\n完成！")
    print(f"  KGChunk 节点总数：{cnt}")
    print(f"  向量索引状态：{idx_state}")


if __name__ == "__main__":
    main()
