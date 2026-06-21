"""Neo4j 知识图谱检索器。

将项目现有的 NER 模型、DeepSeek 意图识别和 Cypher 查询模式封装到
单个 KGRetriever 类中，提供统一的 ``search(query) -> List[Dict]`` 接口。

不重写原有逻辑 —— NER 从 ``ner_model`` 导入，
意图识别复用已有的意图识别提示词，
Cypher 模式镜像了原 generate_prompt 阶段的查询逻辑。

长文本属性（疾病病因/预防措施/疾病简介）走 HAS_CHUNK + 余弦召回，
其余短属性和关系查询保持原有 Cypher 逻辑不变。
"""

from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Optional

try:
    import py2neo
except Exception:  # pragma: no cover - optional Neo4j dependency
    py2neo = None  # type: ignore[assignment]

from medrag.config.settings import settings
from medrag.llm import get_llm_client
from medrag.retrieval.intent import extract_focus, recognize_intents
try:
    from medrag.ner import model as zwk
except Exception:  # pragma: no cover - optional NER runtime dependency
    zwk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 意图 → Cypher 映射
# ---------------------------------------------------------------------------
# 每个元组: (keyword, query_type, relation_or_attribute, target_type, required_entity_type)
# query_type: "attribute" | "chunk_attribute" | "relation" | "reverse_relation"
#
# chunk_attribute: 属性值较长，走 HAS_CHUNK + 余弦召回
# attribute:       属性值较短，直接 RETURN a.<attr>
#
# 注意：keyword 顺序很重要。"治疗周期" 必须在 "治疗" 之前检查以避免
# 错误匹配；"查询疾病所属科目" 是整句级别的检查
# ---------------------------------------------------------------------------
_INTENT_SPEC: List[tuple] = [
    ("简介",           "chunk_attribute",  "疾病简介",       None,       "疾病"),
    ("病因",           "chunk_attribute",  "疾病病因",       None,       "疾病"),
    ("预防",           "chunk_attribute",  "预防措施",       None,       "疾病"),
    ("治疗周期",       "attribute",        "治疗周期",       None,       "疾病"),
    ("治愈概率",       "attribute",        "治愈概率",       None,       "疾病"),
    ("易感人群",       "attribute",        "疾病易感人群",   None,       "疾病"),
    ("药品",           "relation",         "疾病使用药品",   "药品",     "疾病"),
    ("宜吃食物",       "relation",         "疾病宜吃食物",   "食物",     "疾病"),
    ("忌吃食物",       "relation",         "疾病忌吃食物",   "食物",     "疾病"),
    ("检查项目",       "relation",         "疾病所需检查",   "检查项目", "疾病"),
    ("查询疾病所属科目", "relation",        "疾病所属科目",   "科目",     "疾病"),
    ("症状",           "relation",         "疾病的症状",     "疾病症状", "疾病"),
    ("治疗",           "relation",         "治疗的方法",     "治疗方法", "疾病"),
    ("并发",           "relation",         "疾病并发疾病",   "疾病",     "疾病"),
    ("生产商",         "reverse_relation", "生产",           "药品商",   "药品"),
]

# chunk_attribute 对应的 Neo4j 属性名（与 KGChunk.attribute 字段对应）
_CHUNK_ATTR_MAP = {
    "疾病简介": "疾病简介",
    "疾病病因": "疾病病因",
    "预防措施": "预防措施",
}

# 各长属性的「属性同义词」：从 query 中剥离掉它们 + 实体名 + 停用词后，
# 若仍有实质残留，说明用户问的是某个具体细分点（细化型），否则是概括型。
_ATTR_SYNONYMS = {
    "疾病病因": ["疾病病因", "病因", "成因", "原因", "诱因", "怎么得", "为什么会",
               "为什么", "如何引起", "引起", "导致", "怎么会", "怎么引起"],
    "预防措施": ["预防措施", "预防", "怎么预防", "如何预防", "怎么防", "如何避免",
               "怎样避免", "防止", "避免"],
    "疾病简介": ["疾病简介", "简介", "介绍", "是什么病", "是什么", "什么病",
               "什么是", "是种什么"],
}

# 停用词/标点/疑问词：残差判定前清洗（疑问词剥掉，避免"什么/哪些"这类
# 纯概括问法残留 2 字、白白触发一次 LLM focus 调用）
_STOP_RE = re.compile(
    r"[的了吗呢啊？?。，,、；;和与及跟是有会要这那一个请问"
    r"什么哪些几谁怎样如何多少呀嘛\s]+"
)

CHUNK_TOP_K = 3  # 细化型每次从 detail chunk 中召回的段落数


class KGRetriever:
    """Neo4j 医学知识图谱统一检索器。

    依赖**现有**的 NER 流水线（``ner_model``），复用了已有的 Cypher 查询模式。
    意图识别使用 DeepSeek 配合已有的 few-shot 意图识别提示词。

    长文本属性（疾病病因/预防措施/疾病简介）通过 HAS_CHUNK 关系 + 余弦相似度
    召回最相关的 top-k 段落，避免将万字原文直接塞入上下文。

    用法::

        retriever = KGRetriever(
            bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag,
        )
        results = retriever.search("糖尿病的病因是什么？")
        # results 为 List[Dict]，每个字典包含:
        #   source, intent, entity, relation, answer, evidence, score
    """

    def __init__(
        self,
        bert_model,
        bert_tokenizer,
        rule,
        tfidf_r,
        device,
        idx2tag,
        neo4j_client: Optional[py2neo.Graph] = None,
        llm_client=None,
        embedding_model=None,
    ):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.rule = rule
        self.tfidf_r = tfidf_r
        self.device = device
        self.idx2tag = idx2tag

        self.neo4j = neo4j_client or self._create_neo4j_client()
        self.llm = llm_client or get_llm_client()
        self._embedder = embedding_model or self._create_embedding_model()

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _create_neo4j_client():
        if py2neo is None:
            return None
        try:
            return py2neo.Graph(
                settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
                name=settings.neo4j_database,
            )
        except Exception as exc:
            logger.warning("Neo4j unavailable (%s), KGRetriever will return empty results", exc)
            return None

    @staticmethod
    def _create_embedding_model():
        try:
            from medrag.vectors.embedding import EmbeddingModel
            return EmbeddingModel()
        except Exception as exc:
            logger.warning("EmbeddingModel unavailable (%s), chunk retrieval will fall back to full text", exc)
            return None

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    def _get_entities(self, query: str) -> Dict[str, str]:
        """NER 流水线: {entity_type: canonical_name}。"""
        try:
            if zwk is None:
                return {}
            return zwk.get_ner_result(
                self.bert_model,
                self.bert_tokenizer,
                query,
                self.rule,
                self.tfidf_r,
                self.device,
                self.idx2tag,
            )
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Cypher 查询：短属性 / 关系（原有逻辑不变）
    # ------------------------------------------------------------------

    def _query_attribute(self, entity: str, attribute: str) -> Optional[str]:
        """``MATCH (a:疾病{{名称:'...'}}) RETURN a.<attribute>``"""
        cypher = "match (a:疾病{名称:'%s'}) return a.%s" % (entity, attribute)
        try:
            row = self.neo4j.run(cypher).data()[0]
            values = list(row.values())
            if values:
                return "".join(str(v) for v in values if v)
        except Exception:
            pass
        return None

    def _query_relation(
        self, entity: str, relation: str, target_type: str
    ) -> Optional[List[str]]:
        """``MATCH (a:疾病{{名称:'...'}})-[r:REL]->(b:TYPE) RETURN b.名称``"""
        cypher = (
            "match (a:疾病{名称:'%s'})-[r:%s]->(b:%s) return b.名称"
            % (entity, relation, target_type)
        )
        try:
            rows = self.neo4j.run(cypher).data()
            return [list(r.values())[0] for r in rows if r.values()]
        except Exception:
            pass
        return None

    def _query_reverse_relation(
        self, entity: str, relation: str, source_type: str
    ) -> Optional[List[str]]:
        """``MATCH (a:SOURCE)-[r:REL]->(b:药品{{名称:'...'}}) RETURN a.名称``"""
        cypher = (
            "match (a:%s)-[r:%s]->(b:药品{名称:'%s'}) return a.名称"
            % (source_type, relation, entity)
        )
        try:
            rows = self.neo4j.run(cypher).data()
            return [list(r.values())[0] for r in rows if r.values()]
        except Exception:
            pass
        return None

    def _resolve_disease_from_symptom(self, symptom: str) -> Optional[str]:
        """反向查找：症状 → 可能的疾病，随机选取一个。"""
        cypher = (
            "match (a:疾病)-[r:疾病的症状]->(b:疾病症状 {名称:'%s'}) return a.名称"
            % symptom
        )
        try:
            rows = self.neo4j.run(cypher).data()
            names = [list(r.values())[0] for r in rows if r.values()]
            if names:
                return random.choice(names)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # 细分点（focus）判定：区分「概括型」与「细化型」query
    # ------------------------------------------------------------------

    def _residual(self, query: str, entity: str, attribute: str) -> str:
        """扣掉实体名 + 属性同义词 + 停用词后，query 残留的「具体限定词」。"""
        r = query.replace(entity, "")
        for syn in _ATTR_SYNONYMS.get(attribute, []):
            r = r.replace(syn, "")
        return _STOP_RE.sub("", r)

    def _resolve_focus(self, query: str, entity: str, attribute: str) -> str:
        """判定用户问的是整体还是某个细分点。

        Returns
        -------
        str
            ``""`` 表示概括型（应返回提纲摘要）；非空表示细化型，返回用于
            余弦召回的细分点关键词。

        策略：先用正则残差做廉价闸门 —— 残差为空即概括型，**不调 LLM**；
        残差非空（可能含细分点）时调 ``extract_focus`` 精确抽取，LLM 失败
        则回退到正则残差本身。
        """
        residual = self._residual(query, entity, attribute)
        if len(residual) < 2:
            return ""  # 清晰的概括型，省去 LLM 调用

        focus = extract_focus(query, attribute)
        if focus is None:          # LLM 调用失败 → 正则兜底
            return residual
        return focus               # "" 表示 LLM 判定为概括型

    # ------------------------------------------------------------------
    # Chunk 召回：概括型取提纲摘要，细化型在 detail 块上余弦
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)

    def _fetch_summary(self, entity: str, attribute: str) -> Optional[str]:
        """取概括型用的提纲摘要（level='summary'）。"""
        cypher = """
            MATCH (d:疾病 {名称: $entity})-[:HAS_CHUNK]->(c:KGChunk {attribute: $attribute})
            WHERE c.level = 'summary'
            RETURN c.text AS text
            LIMIT 1
        """
        try:
            rows = self.neo4j.run(cypher, entity=entity, attribute=attribute).data()
            if rows:
                return rows[0]["text"]
        except Exception:
            pass
        return None

    def _query_chunks(
        self,
        entity: str,
        attribute: str,
        query: str,
        top_k: int = CHUNK_TOP_K,
    ) -> Optional[List[str]]:
        """长属性两层召回。

        - **概括型**（"糖尿病的病因是什么"）→ 返回离线提纲摘要，长度受控；
        - **细化型**（"糖尿病和遗传有关吗"）→ 用细分点关键词在 detail 块上
          余弦召回 top_k 段。

        各分支不可用时逐级降级：摘要缺失 → 全部 detail 块；detail 缺失或无
        embedding → 直接查原始属性值。
        """
        if self.neo4j is None:
            return None

        focus = self._resolve_focus(query, entity, attribute)

        # ---------- 概括型：提纲摘要 ----------
        if not focus:
            summary = self._fetch_summary(entity, attribute)
            if summary:
                return [summary]
            # 无摘要（如旧数据未重建）→ 降级返回全部 detail 块
            return self._fetch_all_details(entity, attribute) \
                or self._fallback_attr(entity, attribute)

        # ---------- 细化型：detail 块余弦召回 ----------
        query_emb: Optional[List[float]] = None
        if self._embedder is not None:
            try:
                query_emb = self._embedder.encode_one(focus, is_query=True)
            except Exception:
                pass

        cypher = """
            MATCH (d:疾病 {名称: $entity})-[:HAS_CHUNK]->(c:KGChunk {attribute: $attribute})
            WHERE c.level = 'detail' OR c.level IS NULL
            RETURN c.text AS text, c.order AS ord, c.embedding AS emb
            ORDER BY c.order
        """
        try:
            rows = self.neo4j.run(cypher, entity=entity, attribute=attribute).data()
        except Exception:
            rows = []

        if not rows:
            return self._fallback_attr(entity, attribute)

        if query_emb:
            rows.sort(key=lambda r: self._cosine(query_emb, r["emb"]), reverse=True)
        return [r["text"] for r in rows[:top_k]]

    def _fetch_all_details(self, entity: str, attribute: str) -> Optional[List[str]]:
        """按原文顺序返回全部 detail 块（摘要缺失时的降级路径）。"""
        cypher = """
            MATCH (d:疾病 {名称: $entity})-[:HAS_CHUNK]->(c:KGChunk {attribute: $attribute})
            WHERE c.level = 'detail' OR c.level IS NULL
            RETURN c.text AS text
            ORDER BY c.order
        """
        try:
            rows = self.neo4j.run(cypher, entity=entity, attribute=attribute).data()
            if rows:
                return [r["text"] for r in rows]
        except Exception:
            pass
        return None

    def _fallback_attr(self, entity: str, attribute: str) -> Optional[List[str]]:
        """最终降级：直接查原始属性全文。"""
        val = self._query_attribute(entity, attribute)
        return [val] if val else None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        intents: Optional[str] = None,
    ) -> List[Dict]:
        """在 Neo4j 知识图谱中搜索与 *query* 相关的信息。

        Parameters
        ----------
        query:
            自然语言医学问题。
        intents:
            原始意图识别结果。为 *None*（默认）时，通过 DeepSeek
            自动检测意图。可传入预计算结果以跳过 LLM 调用。

        Returns
        -------
        List[Dict]
            每个字典包含键 ``source``、``intent``、``entity``、
            ``relation``、``answer``、``evidence``、``score``。
            当未找到实体或所有查询返回空时，返回空列表。
        """
        # 1. 命名实体识别
        entities = self._get_entities(query)
        if not entities:
            return []

        # 2. 意图识别（若调用方提供则使用缓存）
        raw_intents = intents if intents is not None else recognize_intents(query, self.llm)
        if not raw_intents:
            return []

        # 3. 特殊情况：仅有症状无疾病 → 反向查找
        disease = entities.get("疾病")
        if "疾病症状" in entities and disease is None:
            disease = self._resolve_disease_from_symptom(entities["疾病症状"])

        # 4. 执行匹配到的意图
        results: List[Dict] = []
        for keyword, qtype, rel_attr, target, req_entity in _INTENT_SPEC:
            if keyword not in raw_intents:
                continue

            entity = entities.get(req_entity)
            if entity is None:
                continue

            evidence = None
            answer_parts: List[str] = []

            if qtype == "chunk_attribute":
                # 长文本：Cypher 锁定实体 → 余弦召回 top-k Chunk
                chunks = self._query_chunks(entity, rel_attr, query)
                if chunks:
                    evidence = chunks
                    header = f"{entity} [{rel_attr}]:\n"
                    answer_parts.append(header + "\n---\n".join(chunks))

            elif qtype == "attribute":
                # 短文本：直接返回属性值
                value = self._query_attribute(entity, rel_attr)
                if value:
                    evidence = value
                    answer_parts.append(f"{entity} [{rel_attr}]: {value}")

            elif qtype == "relation":
                names = self._query_relation(entity, rel_attr, target)  # type: ignore[arg-type]
                if names:
                    evidence = names
                    answer_parts.append(
                        f"{entity} [{rel_attr}] → {'、'.join(names)}"
                    )

            elif qtype == "reverse_relation":
                names = self._query_reverse_relation(entity, rel_attr, target)  # type: ignore[arg-type]
                if names:
                    evidence = names
                    answer_parts.append(
                        f"{'、'.join(names)} [{rel_attr}] → {entity}"
                    )

            if evidence is not None:
                results.append(
                    {
                        "source": "neo4j_kg",
                        "intent": keyword,
                        "entity": entity,
                        "relation": rel_attr,
                        "answer": "".join(answer_parts),
                        "evidence": evidence,
                        "score": 1.0,
                    }
                )

        return results
