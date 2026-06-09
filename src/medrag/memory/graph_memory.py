"""Graph-enhanced memory — wraps LongTermMemory with Neo4j relationship edges.

Mirrors AGI-saber internal/memory/graph_memory.go.

When Neo4j is available:
  - Creates memory nodes and FOLLOWS/SIMILAR_TO edges for associative expansion.
  - GraphAwareConsolidate protects high-centrality nodes from deletion.
When Neo4j is unavailable:
  - Transparently falls back to plain LongTermMemory.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .long_term import LongTermMemory
from .types import ConsolidationResult, MemoryItem, RecallFilter

logger = logging.getLogger(__name__)


class GraphMemory:
    """LongTermMemory wrapper with optional Neo4j graph layer.

    Usage::

        ltm = LongTermMemory()
        gm = GraphMemory(ltm, kg_store=None)  # pure LTM, no graph
        gm.store_classified("患者对青霉素过敏", 0.9, embedding, "fact", ["medical"])
        results = gm.recall("过敏")
    """

    def __init__(self, ltm: LongTermMemory,
                 kg_store: Optional[object] = None,
                 sim_threshold: float = 0.7):
        """
        Args:
            ltm: The underlying LongTermMemory instance.
            kg_store: Optional Neo4j/KG store. Must have methods:
                available(), upsert_memory_node(), add_memory_edge(),
                expand_memory_neighbors(), get_high_centrality_memory_ids(),
                delete_memory_node(). If None or unavailable, gracefully degrades.
            sim_threshold: Cosine threshold for automatic SIMILAR_TO edges.
        """
        self._ltm = ltm
        self._kg = kg_store
        self._sim_thresh = sim_threshold
        self._prev_id: int = -1

    @property
    def ltm(self) -> LongTermMemory:
        return self._ltm

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, content: str, importance: float = 0.5,
              embedding: Optional[np.ndarray] = None) -> tuple[bool, int]:
        """Store unclassified. Returns (is_new, item_id)."""
        return self.store_classified(content, importance, embedding, "general", [], "")

    def store_classified(self, content: str, importance: float = 0.5,
                         embedding: Optional[np.ndarray] = None,
                         category: str = "general",
                         tags: Optional[List[str]] = None,
                         slot_hint: str = "") -> tuple[bool, int]:
        """Store with classification. Graph operations run async when available.

        Returns (is_new, item_id). item_id is -1 if store failed.

        Mirrors AGI-saber GraphMemory.StoreClassified().
        """
        tags = tags or []
        added = self._ltm.store_classified(content, importance, embedding, category, tags, slot_hint)

        if not added:
            # Deduped — find the existing item's ID
            return False, self._find_most_similar_id(embedding)

        if not self._ltm.items:
            return True, -1

        new_item = self._ltm.items[-1]
        new_id = new_item.id

        # ── Optional graph layer ──
        if self._kg is not None and self._is_graph_available():
            try:
                self._kg.upsert_memory_node(new_id, content, importance)
                if self._prev_id >= 0:
                    self._kg.add_memory_edge(self._prev_id, new_id, "FOLLOWS", 1.0)
                self._link_similar_edges(new_item, new_id)
            except Exception:
                logger.debug("Graph memory ops failed (non-fatal)", exc_info=True)

        self._prev_id = new_id
        return True, new_id

    def store_item(self, item: MemoryItem) -> None:
        """Direct insert from DB restore.

        Mirrors AGI-saber GraphMemory.StoreItem().
        """
        self._ltm.store_item(item)
        if self._kg is not None and self._is_graph_available():
            try:
                self._kg.upsert_memory_node(item.id, item.content, item.importance)
            except Exception:
                logger.debug("Graph upsert on restore failed", exc_info=True)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(self, query: str = "", top_k: int = 5,
               query_embedding: Optional[np.ndarray] = None) -> List[MemoryItem]:
        """Simple recall with default threshold, then graph expansion.

        Mirrors AGI-saber GraphMemory.Recall().
        """
        return self.recall_by_filter(
            query, query_embedding,
            RecallFilter(top_k=top_k, min_score=0.4),
        )

    def recall_by_filter(self, query: str = "",
                         query_embedding: Optional[np.ndarray] = None,
                         filter: Optional[RecallFilter] = None) -> List[MemoryItem]:
        """Schema-driven recall: semantic recall + optional graph expansion.

        First does LTM recall_by_filter, then if graph is available,
        expands via neighbor edges and appends non-duplicate results.

        Mirrors AGI-saber GraphMemory.RecallByFilter().
        """
        seed_items = self._ltm.recall_by_filter(query, query_embedding, filter)

        if (self._kg is None or not self._is_graph_available() or not seed_items):
            return seed_items

        # ── Graph expansion ──
        try:
            seed_ids = [item.id for item in seed_items]
            expanded_ids = self._kg.expand_memory_neighbors(seed_ids, hops=1)
            if not expanded_ids:
                return seed_items

            id_set = {item.id for item in seed_items}
            expanded: List[MemoryItem] = []
            for eid in expanded_ids:
                if eid in id_set:
                    continue
                for item in self._ltm.items:
                    if item.id == eid:
                        if filter and filter.categories and item.category not in filter.categories:
                            continue
                        item.score = 0.45  # graph expansion base score
                        expanded.append(item)
                        id_set.add(eid)
                        break

            if not expanded:
                return seed_items

            # Merge and re-sort
            all_items = seed_items + expanded
            all_items.sort(key=lambda x: x.score, reverse=True)
            if filter and filter.top_k > 0 and len(all_items) > filter.top_k:
                all_items = all_items[:filter.top_k]
            return all_items
        except Exception:
            logger.debug("Graph recall expansion failed (non-fatal)", exc_info=True)
            return seed_items

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def graph_aware_consolidate(self) -> ConsolidationResult:
        """Consolidate with graph-aware protection of high-centrality nodes.

        Mirrors AGI-saber GraphMemory.GraphAwareConsolidate().
        """
        result = self._ltm.consolidate()

        if self._kg is None or not self._is_graph_available():
            return result

        # Protect high-centrality nodes (in-degree >= 3)
        try:
            protected = self._kg.get_high_centrality_memory_ids(result.deleted_ids, threshold=3)
            if protected:
                protect_set = set(protected)
                filtered = [did for did in result.deleted_ids if did not in protect_set]
                logger.info(
                    "Graph centrality: %d memories spared (in-degree>=3)",
                    len(result.deleted_ids) - len(filtered),
                )
                result.deleted_ids = filtered
        except Exception:
            logger.debug("Graph centrality check failed (non-fatal)", exc_info=True)

        # Sync deletions to graph
        try:
            for mem_id in result.deleted_ids:
                self._kg.delete_memory_node(mem_id)
        except Exception:
            logger.debug("Graph deletion sync failed (non-fatal)", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_graph_available(self) -> bool:
        if self._kg is None:
            return False
        available = getattr(self._kg, "available", None)
        if callable(available):
            return available()
        return bool(getattr(self._kg, "available", False))

    def _link_similar_edges(self, new_item: MemoryItem, new_id: int) -> None:
        """Scan recent items and create SIMILAR_TO edges.

        Mirrors AGI-saber GraphMemory.linkSimilarEdges().
        """
        items = self._ltm.items
        start = max(0, len(items) - 51)
        for i in range(start, len(items) - 1):
            old = items[i]
            if old.id == new_id:
                continue
            if old.embedding is None or new_item.embedding is None:
                continue
            from .long_term import _cosine
            sim = _cosine(old.embedding, new_item.embedding)
            if sim >= self._sim_thresh:
                self._kg.add_memory_edge(old.id, new_id, "SIMILAR_TO", sim)

    def _find_most_similar_id(self, embedding: Optional[np.ndarray]) -> int:
        """Find the item ID most similar to *embedding* (for dedup return).

        Mirrors AGI-saber GraphMemory.findMostSimilarID().
        """
        if embedding is None or not self._ltm.items:
            return -1
        best_id, best_sim = -1, 0.0
        from .long_term import _cosine
        for item in self._ltm.items:
            if item.embedding is None or len(item.embedding) != len(embedding):
                continue
            s = _cosine(embedding, item.embedding)
            if s > best_sim:
                best_sim, best_id = s, item.id
        return best_id

    # ------------------------------------------------------------------
     # Proxies
    # ------------------------------------------------------------------

    def need_consolidation(self) -> bool:
        return self._ltm.need_consolidation()

    def set_consolidation_config(self, cfg) -> None:
        self._ltm.set_consolidation_config(cfg)

    def sync_prev_id(self) -> None:
        if self._ltm.items:
            self._prev_id = self._ltm.items[-1].id

    def __len__(self) -> int:
        return self._ltm.count()
