"""Neo4j Knowledge Graph Retriever.

Wraps the project's existing NER model, DeepSeek intent recognition, and
Cypher query patterns into a single KGRetriever class with a unified
``search(query) -> List[Dict]`` interface.

Does NOT rewrite the original logic — NER is imported from ``ner_model``,
intent recognition reuses the same prompt as ``webui.Intent_Recognition``,
and Cypher patterns mirror ``webui.generate_prompt``.
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
# Intent → Cypher mapping
# ---------------------------------------------------------------------------
# Each tuple: (keyword, query_type, relation_or_attribute, target_type, required_entity_type)
# query_type: "attribute" | "relation" | "reverse_relation"
#
# NOTE: keyword order matters. "治疗周期" must be checked before "治疗" to
# avoid false matches; "查询疾病所属科目" is a full sentence-level check
# (see webui.py:217).  This list mirrors the if/elif chain in
# webui.generate_prompt exactly.
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
    """Unified retriever over the Neo4j medical knowledge graph.

    Relies on the **existing** NER pipeline (``ner_model``) and replicates
    the Cypher patterns from ``webui.generate_prompt``.  Intent recognition
    uses DeepSeek with the same few-shot prompt as ``webui.Intent_Recognition``.

    Usage::

        # Obtain NER components (same as webui.load_model)
        rule = zwk.rule_find()
        tfidf_r = zwk.tfidf_alignment()
        ...

        retriever = KGRetriever(
            bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag,
        )
        results = retriever.search("感冒了怎么办")
        # results is List[Dict], each dict has:
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
    # Internal helpers
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
        """NER pipeline: {entity_type: canonical_name}."""
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
    # Intent recognition
    # ------------------------------------------------------------------
    # The prompt below is copied from webui.Intent_Recognition to avoid
    # importing Streamlit-dependent modules.  If you extract intent
    # recognition into a shared utility, import it here instead.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cypher query helpers
    # ------------------------------------------------------------------
    # These replicate the add_shuxing_prompt / add_lianxi_prompt functions
    # from webui.py but return raw data instead of prompt strings.
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
        """Reverse-lookup: symptom → possible diseases, pick one at random."""
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
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        intents: Optional[str] = None,
    ) -> List[Dict]:
        """Search the Neo4j knowledge graph for information relevant to *query*.

        Parameters
        ----------
        query:
            Natural-language medical question.
        intents:
            Raw intent recognition result.  When *None* (default), intents
            are auto-detected via DeepSeek.  You can pass a pre-computed
            string (the output of ``Intent_Recognition``) to skip the LLM
            call.

        Returns
        -------
        List[Dict]
            Each dict has keys ``source``, ``intent``, ``entity``,
            ``relation``, ``answer``, ``evidence``, ``score``.
            Returns an empty list when no entity is found or all queries
            come back empty.
        """
        # 1. NER
        entities = self._get_entities(query)
        if not entities:
            return []

        # 2. Intent recognition (cached if caller provided it)
        raw_intents = intents if intents is not None else recognize_intents(query, self.llm)
        if not raw_intents:
            return []

        # 3. Special case: symptom without disease → reverse-lookup
        disease = entities.get("疾病")
        if "疾病症状" in entities and disease is None:
            disease = self._resolve_disease_from_symptom(entities["疾病症状"])

        # 4. Execute matched intents
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
