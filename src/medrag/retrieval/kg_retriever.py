"""Neo4j 知识图谱检索器。

将项目现有的 NER 模型、DeepSeek 意图识别和 Cypher 查询模式封装到
单个 KGRetriever 类中，提供统一的 ``search(query) -> List[Dict]`` 接口。

不重写原有逻辑 —— NER 从 ``ner_model`` 导入，
意图识别复用与 ``webui.Intent_Recognition`` 相同的提示词，
Cypher 模式镜像了 ``webui.generate_prompt`` 的实现。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import py2neo

from medrag.config.settings import settings
from medrag.llm import get_llm_client
from medrag.retrieval.intent import recognize_intents
from medrag.ner import model as zwk

# ---------------------------------------------------------------------------
# 意图 → Cypher 映射
# ---------------------------------------------------------------------------
# 每个元组: (keyword, query_type, relation_or_attribute, target_type, required_entity_type)
# query_type: "attribute" | "relation" | "reverse_relation"
#
# 注意：keyword 顺序很重要。"治疗周期" 必须在 "治疗" 之前检查以避免
# 错误匹配；"查询疾病所属科目" 是整句级别的检查
# （参见 webui.py:217）。此列表完全镜像了 webui.generate_prompt
# 中的 if/elif 链。
# ---------------------------------------------------------------------------
_INTENT_SPEC: List[tuple] = [
    ("简介",           "attribute",        "疾病简介",       None,       "疾病"),
    ("病因",           "attribute",        "疾病病因",       None,       "疾病"),
    ("预防",           "attribute",        "预防措施",       None,       "疾病"),
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


class KGRetriever:
    """Neo4j 医学知识图谱统一检索器。

    依赖**现有**的 NER 流水线（``ner_model``），复制了 ``webui.generate_prompt``
    中的 Cypher 模式。意图识别使用 DeepSeek 配合与 ``webui.Intent_Recognition``
    相同的 few-shot 提示词。

    用法::

        # 获取 NER 组件（与 webui.load_model 相同）
        rule = zwk.rule_find()
        tfidf_r = zwk.tfidf_alignment()
        ...

        retriever = KGRetriever(
            bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag,
        )
        results = retriever.search("感冒了怎么办")
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
    ):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.rule = rule
        self.tfidf_r = tfidf_r
        self.device = device
        self.idx2tag = idx2tag

        self.neo4j = neo4j_client or self._create_neo4j_client()

        self.llm = get_llm_client("deepseek")

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _create_neo4j_client() -> py2neo.Graph:
        return py2neo.Graph(
            settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            name=settings.neo4j_database,
        )

    def _get_entities(self, query: str) -> Dict[str, str]:
        """NER 流水线: {entity_type: canonical_name}。"""
        try:
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
    # 意图识别
    # ------------------------------------------------------------------
    # 以下提示词从 webui.Intent_Recognition 复制而来，以避免导入
    # 依赖 Streamlit 的模块。如果将意图识别提取到共享工具模块中，
    # 改从这里导入。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cypher 查询辅助方法
    # ------------------------------------------------------------------
    # 这些方法复制了 webui.py 中的 add_shuxing_prompt / add_lianxi_prompt
    # 函数，但返回原始数据而非提示字符串。
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
            自动检测意图。可传入预计算结果
            （``Intent_Recognition`` 的输出）以跳过 LLM 调用。

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

            if qtype == "attribute":
                value = self._query_attribute(entity, rel_attr)
                if value:
                    evidence = value
                    answer_parts.append(value)

            elif qtype == "relation":
                names = self._query_relation(entity, rel_attr, target)  # type: ignore[arg-type]
                if names:
                    evidence = names
                    answer_parts.append("、".join(names))

            elif qtype == "reverse_relation":
                names = self._query_reverse_relation(entity, rel_attr, target)  # type: ignore[arg-type]
                if names:
                    evidence = names
                    answer_parts.append("、".join(names))

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
