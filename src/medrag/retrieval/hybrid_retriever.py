"""混合多源检索器。

通过 QueryRouter 编排三个检索后端：
  1. KGRetriever         — Neo4j 医学知识图谱（独立传递，不参与 RRF/重排）
  2. Milvus ANN           — BGE dense 语义召回（cMedQA2 语料库）
  3. ES BM25              — 关键词 / 医学实体召回（cMedQA2 语料库）

cMedQA2 语料库采用双路检索 + RRF 融合 + Cross-Encoder 精选：
  Milvus (dense) + ES (sparse) → RRF (c=60) → Cross-Encoder → top-k
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RRF_C = 60


def _rrf_fuse(
    results_a: List[Dict],
    results_b: List[Dict],
    tag_a: str = "dense",
    tag_b: str = "sparse",
) -> List[Dict]:
    """对**同一语料库**的两路检索结果进行倒数排名融合（跨源叠加版）。

    标准 RRF: score(d) = Σ 1 / (RRF_C + rank_s(d))
    同一文档在两路中都出现时，分数累加，实现跨源贡献叠加。

    Args:
        results_a: 第一路检索结果（按 score 降序，已排名）。
        results_b: 第二路检索结果（按 score 降序，已排名）。
        tag_a: results_a 的来源标记。
        tag_b: results_b 的来源标记。

    Returns:
        按 RRF 总分降序排列的融合结果列表。
    """
    score_map: Dict[str, dict] = {}

    def _accumulate(results, tag):
        for rank, r in enumerate(results, start=1):
            doc_id = r.get("id")
            contrib = 1.0 / (RRF_C + rank)
            if doc_id in score_map:
                score_map[doc_id]["score"] += contrib
                existing = score_map[doc_id]["result"]
                existing["rrf_source"] = f"{existing.get('rrf_source', '')}+{tag}"
                existing["rrf_source_rank"] = (
                    f"{existing.get('rrf_source_rank', '')}+{rank}"
                )
            else:
                entry = dict(r)
                entry["rrf_source"] = tag
                entry["rrf_source_rank"] = str(rank)
                score_map[doc_id] = {"result": entry, "score": contrib}

    _accumulate(results_a, tag_a)
    _accumulate(results_b, tag_b)

    sorted_items = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    for item in sorted_items:
        item["result"]["rrf_score"] = round(item["score"], 6)

    return [item["result"] for item in sorted_items]


def _tag_single_source(results: List[Dict], source_tag: str) -> List[Dict]:
    """为单源结果附加排名标记（无融合）。"""
    tagged: List[Dict] = []
    for rank, r in enumerate(results, start=1):
        r = dict(r)
        r["rrf_score"] = None
        r["rrf_source"] = source_tag
        r["rrf_source_rank"] = rank
        tagged.append(r)
    return tagged


class HybridRetriever:
    """带路由的统一多源检索。

    KG 结果独立传递；cMedQA2 语料库使用 Dense + Sparse 双路 RRF 融合。
    Cross-Encoder 重排由下游 chat_service 在 QA 结果上单独执行。

    用法::

        hybrid = HybridRetriever(kg=kg, milvus=qa, es=es, router=router)
        result = hybrid.retrieve("感冒了怎么办")
    """

    def __init__(
        self,
        kg_retriever=None,
        qa_retriever=None,
        es_retriever=None,
        router=None,
        case_retriever=None,
        normalizer=None,
    ):
        self.kg = kg_retriever
        self.qa = qa_retriever
        self.es = es_retriever
        self.router = router
        self.case_retriever = case_retriever
        self.normalizer = normalizer

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        department: str | None = None,
        username: str | None = None,
        route: dict | None = None,
    ) -> Dict:
        """路由 *query* 并从合适的源获取结果。

        KG 结果独立返回；QA 结果经 Milvus + ES 双路 RRF 融合后返回。

        Returns:
            字典，键为：``route``、``kg_results``、``qa_results``、
            ``qa_source_details``、``case_results``、``fusion_mode``。
        """
        query_info = None
        retrieval_query = query
        if self.normalizer is not None:
            query_info = self.normalizer.normalize(query).to_dict()
            retrieval_query = query_info["normalized_query"]

        # 优先使用调用方已确定的路由，避免重复路由
        if route is None:
            route = getattr(self, '_current_route', None)
        if route is None:
            route = self.router.route(retrieval_query)

        use_kg = route["use_kg"]
        use_qa = route["use_qa"]
        use_case = bool(route.get("needs_case_context")) and self.case_retriever is not None

        # 两者都不开 → 直接返回
        if not use_kg and not use_qa and not use_case:
            return {
                "route": route,
                "query_info": query_info,
                "kg_results": [],
                "qa_results": [],
                "qa_source_details": {"milvus": [], "es": []},
                "case_results": [],
                "fusion_mode": "none",
            }

        # --- 1. KG 检索（独立，不参与 RRF/重排）---
        kg_results = self._safe_kg_search(retrieval_query) if use_kg else []

        # --- 2. QA 双路检索（Milvus + ES → RRF）---
        qa_results, qa_details = self._retrieve_qa(
            retrieval_query, top_k, department,
        ) if use_qa else ([], {"milvus": [], "es": []})

        # --- 3. 用户病例检索（独立）---
        case_results = (
            self._safe_case_search(retrieval_query, username, top_k)
            if use_case else []
        )

        # --- 4. 融合判定 ---
        if qa_results:
            fusion_mode = "rrf_dense_sparse" if (qa_details["milvus"] and qa_details["es"]) else "single"
        else:
            fusion_mode = "none"

        return {
            "route": route,
            "query_info": query_info,
            "kg_results": kg_results,
            "qa_results": qa_results,
            "qa_source_details": qa_details,
            "case_results": case_results,
            "fusion_mode": fusion_mode,
        }

    # ------------------------------------------------------------------
    # QA 双路检索（Dense + Sparse → RRF）
    # ------------------------------------------------------------------

    def _retrieve_qa(
        self, query: str, top_k: int, department: str | None,
    ) -> tuple[List[Dict], Dict[str, List[Dict]]]:
        """cMedQA2 双路检索：Milvus（dense）+ ES（sparse）→ RRF。"""
        milvus_results: List[Dict] = []
        es_results: List[Dict] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._safe_qa_search, query, top_k, department): "milvus",
                executor.submit(self._safe_es_search, query, top_k, department): "es",
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.warning("%s retrieval failed", source, exc_info=True)
                    result = []
                if source == "milvus":
                    milvus_results = result
                else:
                    es_results = result

        details = {"milvus": milvus_results, "es": es_results}

        if milvus_results and es_results:
            return _rrf_fuse(milvus_results, es_results, tag_a="milvus", tag_b="es"), details
        if milvus_results:
            return _tag_single_source(milvus_results, "milvus"), details
        if es_results:
            return _tag_single_source(es_results, "es"), details
        return [], details

    # ------------------------------------------------------------------
    # 内部辅助方法（每个源独立 try/except）
    # ------------------------------------------------------------------

    def _safe_kg_search(self, query: str) -> List[Dict]:
        try:
            return self.kg.search(query)
        except Exception:
            logger.warning("KG retrieval failed", exc_info=True)
            return []

    def _safe_qa_search(self, query: str, top_k: int, department: str | None = None) -> List[Dict]:
        try:
            return self.qa.search(query, top_k=top_k, department=department)
        except Exception:
            logger.warning("Milvus retrieval failed", exc_info=True)
            return []

    def _safe_es_search(self, query: str, top_k: int, department: str | None = None) -> List[Dict]:
        if self.es is None:
            return []
        try:
            return self.es.search(query, top_k=top_k, department=department)
        except Exception:
            logger.warning("ES retrieval failed", exc_info=True)
            return []

    def _safe_case_search(self, query: str, username: str | None, top_k: int) -> List[Dict]:
        try:
            return self.case_retriever.search(query, username=username, top_k=top_k)
        except Exception:
            logger.warning("User case retrieval failed", exc_info=True)
            return []
