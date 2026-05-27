"""混合多源检索器。

通过 QueryRouter 编排两个检索后端：
  1. KGRetriever      — Neo4j 医学知识图谱
  2. ToyhomQARetriever — Milvus 支持的医学问答库

每个源独立调用；一个失败不影响另一个。
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class HybridRetriever:
    """带路由的统一多源检索。

    用法::

        hybrid = HybridRetriever(kg_retriever=kg, toyhom_retriever=toy, router=router)
        result = hybrid.retrieve("感冒了怎么办")
        # result["all_results"] → 来自所有活跃源的合并列表
    """

    def __init__(self, kg_retriever, toyhom_retriever, router):
        self.kg = kg_retriever
        self.toyhom = toyhom_retriever
        self.router = router

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        department: str | None = None,
    ) -> Dict:
        """路由 *query* 并从合适的源获取结果。

        Args:
            query: 自然语言医学问题。
            top_k: 向量库返回的最大结果数。
            department: 可选科室过滤，透传给 ToyhomQARetriever.search()。

        Returns:
            字典，键为：route、kg_results、toyhom_results、all_results。
        """
        route = self.router.route(query)

        kg_results: List[Dict] = []
        toyhom_results: List[Dict] = []

        # --- 知识图谱 ---
        if route["use_kg"]:
            kg_results = self._safe_kg_search(query)

        # --- Toyhom 问答 ---
        if route["use_toyhom_qa"]:
            toyhom_results = self._safe_toyhom_search(query, top_k, department)

        # 合并
        all_results: List[Dict] = kg_results + toyhom_results

        return {
            "route": route,
            "kg_results": kg_results,
            "toyhom_results": toyhom_results,
            "all_results": all_results,
        }

    # ------------------------------------------------------------------
    # 内部辅助方法（每个源独立 try/except）
    # ------------------------------------------------------------------

    def _safe_kg_search(self, query: str) -> List[Dict]:
        try:
            return self.kg.search(query)
        except Exception:
            logger.warning("KG retrieval failed", exc_info=True)
            return []

    def _safe_toyhom_search(self, query: str, top_k: int, department: str | None = None) -> List[Dict]:
        try:
            return self.toyhom.search(query, top_k=top_k, department=department)
        except Exception:
            logger.warning("Toyhom retrieval failed", exc_info=True)
            return []
