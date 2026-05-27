"""Hybrid multi-source retriever.

Orchestrates two retrieval backends via a QueryRouter:
  1. KGRetriever      — Neo4j medical knowledge graph
  2. ToyhomQARetriever — Milvus-backed medical QA

Each source is called independently; one failure does not affect the others.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Unified multi-source retrieval with routing.

    Usage::

        hybrid = HybridRetriever(kg_retriever=kg, toyhom_retriever=toy, router=router)
        result = hybrid.retrieve("感冒了怎么办")
        # result["all_results"] → merged list from all active sources
    """

    def __init__(self, kg_retriever, toyhom_retriever, router):
        self.kg = kg_retriever
        self.toyhom = toyhom_retriever
        self.router = router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """Route *query* and fetch results from the appropriate sources.

        Args:
            query: Natural-language medical question.
            top_k: Max results from the vector store.

        Returns:
            Dict with keys: route, kg_results, toyhom_results, all_results.
        """
        route = self.router.route(query)

        kg_results: List[Dict] = []
        toyhom_results: List[Dict] = []

        # --- KG ---
        if route["use_kg"]:
            kg_results = self._safe_kg_search(query)

        # --- Toyhom QA ---
        if route["use_toyhom_qa"]:
            toyhom_results = self._safe_toyhom_search(query, top_k)

        # Merge
        all_results: List[Dict] = kg_results + toyhom_results

        return {
            "route": route,
            "kg_results": kg_results,
            "toyhom_results": toyhom_results,
            "all_results": all_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers  (isolated try/except per source)
    # ------------------------------------------------------------------

    def _safe_kg_search(self, query: str) -> List[Dict]:
        try:
            return self.kg.search(query)
        except Exception:
            logger.warning("KG retrieval failed", exc_info=True)
            return []

    def _safe_toyhom_search(self, query: str, top_k: int) -> List[Dict]:
        try:
            return self.toyhom.search(query, top_k=top_k)
        except Exception:
            logger.warning("Toyhom retrieval failed", exc_info=True)
            return []
