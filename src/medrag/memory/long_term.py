"""Long-term memory — semantic recall with embedding or TF-IDF fallback.

Mirrors AGI-saber internal/memory/memory.go LongTerm.

Key design:
  - StoreClassified: inline dedup check via embedding cosine, then append.
  - Recall / RecallByFilter: score = sim*0.7 + importance*0.3, threshold gate.
  - FilterByCategory: stable category enumeration (no scoring).
  - Consolidate: decay → dedup+merge → expire, all in-place on self._items.
  - TF-IDF fallback: when no embedding provided, tokenize by Chinese chars + English words.
  - Persistence: PostgreSQL-backed via postgres_client (``username`` for multi-tenant isolation).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

import numpy as np

from .types import (
    ConsolidationConfig,
    ConsolidationResult,
    MemoryItem,
    RecallFilter,
)

# ---------------------------------------------------------------------------
# Tokenization & similarity
# ---------------------------------------------------------------------------

_CHINESE_RE = re.compile(r"[一-鿿]")
_ENGLISH_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Chinese char + English word tokenizer.

    Mirrors AGI-saber tokenize() in memory.go.
    """
    tokens: List[str] = []
    for word in _ENGLISH_RE.findall(text):
        tokens.append(word.lower())
    for char in _CHINESE_RE.findall(text):
        tokens.append(char)
    return tokens


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Mirrors AGI-saber cosine() in memory.go.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _merge_tags(a: List[str], b: List[str]) -> List[str]:
    """Merge two tag lists deduplicated, preserving order.

    Mirrors AGI-saber mergeTags() in memory.go.
    """
    seen: set = set()
    result: List[str] = []
    for t in a + b:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """Embedding-based semantic recall with TF-IDF fallback and automatic consolidation.

    Replaces JSON file persistence with PostgreSQL.
    Supports optional ``username`` for multi-tenant isolation.

    Usage::

        ltm = LongTermMemory(username="alice")
        ltm.set_consolidation_config(ConsolidationConfig(...))
        ltm.store_classified("患者对青霉素过敏", 0.9, embedding, "fact", ["medical"])
        results = ltm.recall_by_filter("过敏", query_embedding=emb)
    """

    def __init__(self, username: str = "", persist_path: Optional[str] = None):
        self._username = username
        self._items: List[MemoryItem] = []
        self._vocab_id: Dict[str, int] = {}      # token → index
        self._vocab: List[str] = []               # index → token
        self._next_id: int = 0
        self._store_count: int = 0
        self._consolidation_cfg: Optional[ConsolidationConfig] = None
        self._content_hashes: Set[str] = set()    # Hash-based exact dedup (stage 1)

        # ── Auto-load from PostgreSQL ──
        self._load_from_pg()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def set_consolidation_config(self, cfg: ConsolidationConfig) -> None:
        self._consolidation_cfg = cfg

    @property
    def consolidation_cfg(self) -> Optional[ConsolidationConfig]:
        return self._consolidation_cfg

    # ------------------------------------------------------------------
    # Vocab (TF-IDF fallback)
    # ------------------------------------------------------------------

    def _build_vocab(self, text: str) -> None:
        for t in _tokenize(text):
            if t not in self._vocab_id:
                self._vocab_id[t] = len(self._vocab)
                self._vocab.append(t)

    def _text_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self._vocab_id), dtype=np.float64)
        for t in _tokenize(text):
            idx = self._vocab_id.get(t)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def _rebuild_vocab(self) -> None:
        self._vocab_id.clear()
        self._vocab.clear()
        for item in self._items:
            self._build_vocab(item.content)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(content: str) -> str:
        """SHA256 hex digest of normalized content for exact-match dedup.

        Strips leading/trailing whitespace before hashing so minor formatting
        differences don't defeat the check.
        """
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def store(self, content: str, importance: float = 0.5,
              embedding: Optional[np.ndarray] = None) -> bool:
        """Store unclassified memory. Returns True if new, False if deduped."""
        return self.store_classified(content, importance, embedding, "general", [], "")

    def store_classified(self, content: str, importance: float = 0.5,
                         embedding: Optional[np.ndarray] = None,
                         category: str = "general",
                         tags: Optional[List[str]] = None,
                         slot_hint: str = "") -> bool:
        """Store with classification metadata.

        Two-stage dedup:
          1. Hash-based exact match (O(1)) — catches byte-identical content.
          2. Embedding cosine similarity — catches semantically near-identical content.

        Stage 1 runs unconditionally; stage 2 requires an embedding and a
        configured dedup_threshold.

        Mirrors AGI-saber LongTerm.StoreClassified().
        """
        tags = tags or []

        # ── Stage 1: Hash-based exact dedup ──
        content_hash = self._hash(content)
        if content_hash in self._content_hashes:
            # Find the existing item and update its metadata
            for existing in self._items:
                if self._hash(existing.content) == content_hash:
                    if importance > existing.importance:
                        existing.importance = importance
                    existing.last_accessed = datetime.now()
                    if category and (not existing.category or existing.category == "general"):
                        existing.category = category
                    if slot_hint and not existing.slot_hint:
                        existing.slot_hint = slot_hint
                    if tags:
                        existing.tags = _merge_tags(existing.tags, tags)
                    return False

        # ── Stage 2: Embedding-based semantic dedup ──
        if (self._consolidation_cfg is not None
                and len(self._items) > 0
                and embedding is not None
                and len(embedding) > 0):
            for existing in self._items:
                if existing.embedding is None or len(existing.embedding) != len(embedding):
                    continue
                sim = _cosine(embedding, existing.embedding)
                if sim >= self._consolidation_cfg.dedup_threshold:
                    # Update existing item instead of inserting
                    if importance > existing.importance:
                        existing.importance = importance
                    existing.last_accessed = datetime.now()
                    if category and (not existing.category or existing.category == "general"):
                        existing.category = category
                    if slot_hint and not existing.slot_hint:
                        existing.slot_hint = slot_hint
                    if tags:
                        existing.tags = _merge_tags(existing.tags, tags)
                    return False

        # ── Build vocab for TF fallback ──
        for item in self._items:
            self._build_vocab(item.content)
        self._build_vocab(content)

        now = datetime.now()
        if not category:
            category = "general"
        item = MemoryItem(
            id=self._next_id,
            content=content,
            importance=importance,
            embedding=embedding,
            created_at=now,
            last_accessed=now,
            category=category,
            tags=list(tags),
            slot_hint=slot_hint,
        )
        self._items.append(item)
        self._next_id += 1
        self._store_count += 1
        self._content_hashes.add(content_hash)
        self._auto_save()
        return True

    def store_item(self, item: MemoryItem) -> None:
        """Directly insert a pre-built item (for DB restore).

        Mirrors AGI-saber LongTerm.StoreItem().
        """
        self._build_vocab(item.content)
        if item.id >= self._next_id:
            self._next_id = item.id + 1
        if item.created_at is None:
            item.created_at = datetime.now()
        if item.last_accessed is None:
            item.last_accessed = item.created_at
        self._items.append(item)
        self._content_hashes.add(self._hash(item.content))
        self._auto_save()

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(self, query: str = "", top_k: int = 5,
               query_embedding: Optional[np.ndarray] = None) -> List[MemoryItem]:
        """Simple recall with default threshold.

        Mirrors AGI-saber LongTerm.Recall().
        """
        return self.recall_by_filter(
            query, query_embedding,
            RecallFilter(top_k=top_k, min_score=0.4),
        )

    def recall_by_filter(self, query: str = "",
                         query_embedding: Optional[np.ndarray] = None,
                         filter: Optional[RecallFilter] = None) -> List[MemoryItem]:
        """Recall with Schema-driven filter constraints.

        Score = cosine_sim * 0.7 + importance * 0.3.
        Items below min_score are filtered out (avoids noise injection).
        Supports category/tag/age filtering.

        Mirrors AGI-saber LongTerm.RecallByFilter().
        """
        if not self._items:
            return []

        f = filter or RecallFilter()
        threshold = f.min_score if f.min_score > 0 else 0.4
        now = datetime.now()

        scored: List[tuple[MemoryItem, float]] = []

        for item in self._items:
            # ── Category filter ──
            if f.categories and item.category not in f.categories:
                continue
            # ── Tag filter ──
            if f.require_tags:
                if not all(tag in item.tags for tag in f.require_tags):
                    continue
            # ── Age filter ──
            if f.max_age_hours and item.created_at:
                hours = (now - item.created_at).total_seconds() / 3600
                if hours > f.max_age_hours:
                    continue

            # ── Similarity score ──
            sim = self._compute_similarity(item, query, query_embedding)
            score = sim * 0.7 + item.importance * 0.3

            if score >= threshold:
                item.last_accessed = now
                scored.append((item, score))

        if not scored:
            return []

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        if f.top_k > 0 and len(scored) > f.top_k:
            scored = scored[:f.top_k]

        results = []
        for item, score in scored:
            item.score = score
            results.append(item)
        return results

    def filter_by_category(self, categories: List[str],
                           limit: int = 0) -> List[MemoryItem]:
        """Return all items belonging to one of the given categories (no scoring).

        Used for Profile/structured slot enumeration.
        Mirrors AGI-saber LongTerm.FilterByCategory().
        """
        if not self._items or not categories:
            return []
        results: List[MemoryItem] = []
        for item in self._items:
            if item.category in categories:
                results.append(item)
                if limit > 0 and len(results) >= limit:
                    break
        return results

    def _compute_similarity(self, item: MemoryItem, query: str,
                            query_emb: Optional[np.ndarray]) -> float:
        """Compute similarity between *item* and the query.

        Embedding cosine preferred; falls back to TF vector cosine.
        Mirrors AGI-saber RecallByFilter's similarity logic.
        """
        if (query_emb is not None and len(query_emb) > 0
                and item.embedding is not None and len(item.embedding) == len(query_emb)):
            return _cosine(query_emb, item.embedding)

        # TF-IDF fallback
        self._build_vocab(query)
        qv = self._text_to_vector(query)
        iv = self._text_to_vector(item.content)
        # Align lengths
        max_len = max(len(qv), len(iv))
        if len(qv) < max_len:
            qv = np.pad(qv, (0, max_len - len(qv)))
        if len(iv) < max_len:
            iv = np.pad(iv, (0, max_len - len(iv)))
        return _cosine(qv, iv)

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def need_consolidation(self) -> bool:
        """Check whether a consolidation cycle is due.

        Mirrors AGI-saber LongTerm.NeedConsolidation().
        """
        return (self._consolidation_cfg is not None
                and self._consolidation_cfg.trigger_interval > 0
                and self._store_count >= self._consolidation_cfg.trigger_interval)

    def consolidate(self) -> ConsolidationResult:
        """Run a full consolidation cycle: decay → dedup+merge → expire.

        Mirrors AGI-saber LongTerm.Consolidate().

        Returns a ConsolidationResult with deleted_ids and updated_items
        so the caller can sync persistence.
        """
        result = ConsolidationResult()
        if self._consolidation_cfg is None or len(self._items) <= 1:
            return result

        cfg = self._consolidation_cfg
        self._store_count = 0
        removed: set = set()

        now = datetime.now()

        # ── Phase 1: Importance decay ──
        for item in self._items:
            if item.created_at:
                days = (now - item.created_at).total_seconds() / 86400
                item.importance *= math.pow(cfg.decay_rate, days)

        # ── Phase 2: Dedup + Merge (pairwise) ──
        n = len(self._items)
        for i in range(n):
            if i in removed:
                continue
            for j in range(i + 1, n):
                if j in removed:
                    continue
                sim = self._item_similarity(self._items[i], self._items[j])

                if sim >= cfg.dedup_threshold:
                    # Dedup: keep the one with higher importance
                    if self._items[j].importance >= self._items[i].importance:
                        removed.add(i)
                        result.deduped += 1
                        result.deleted_ids.append(self._items[i].id)
                    else:
                        removed.add(j)
                        result.deduped += 1
                        result.deleted_ids.append(self._items[j].id)
                elif sim >= cfg.similarity_threshold:
                    # Merge: combine similar items
                    merged = self._merge_items(self._items[i], self._items[j])
                    self._items[i] = merged
                    removed.add(j)
                    result.merged += 1
                    result.deleted_ids.append(self._items[j].id)
                    result.updated_items.append(merged)

        # ── Phase 3: Expire ──
        for i in range(n):
            if i in removed:
                continue
            item = self._items[i]
            if item.created_at:
                days = (now - item.created_at).total_seconds() / 86400
                if (cfg.ttl_days > 0
                        and days > cfg.ttl_days
                        and item.importance < cfg.min_importance):
                    removed.add(i)
                    result.expired += 1
                    result.deleted_ids.append(item.id)

        # ── Rebuild items list and vocab ──
        self._items = [item for i, item in enumerate(self._items) if i not in removed]
        self._rebuild_vocab()
        self._content_hashes = {self._hash(item.content) for item in self._items}

        # ── Sync to PostgreSQL ──
        self._sync_consolidation_to_pg(result)

        return result

    def _item_similarity(self, a: MemoryItem, b: MemoryItem) -> float:
        """Cosine similarity between two items (embedding preferred, TF fallback).

        Mirrors AGI-saber LongTerm.itemSimilarity().
        """
        if (a.embedding is not None and b.embedding is not None
                and len(a.embedding) == len(b.embedding)):
            return _cosine(a.embedding, b.embedding)

        # TF fallback
        self._build_vocab(a.content)
        self._build_vocab(b.content)
        av = self._text_to_vector(a.content)
        bv = self._text_to_vector(b.content)
        max_len = max(len(av), len(bv))
        if len(av) < max_len:
            av = np.pad(av, (0, max_len - len(av)))
        if len(bv) < max_len:
            bv = np.pad(bv, (0, max_len - len(bv)))
        return _cosine(av, bv)

    @staticmethod
    def _merge_items(a: MemoryItem, b: MemoryItem) -> MemoryItem:
        """Merge two similar items, preserving the more important one as base.

        Mirrors AGI-saber LongTerm.mergeItems().

        - Content: if neither is a substring of the other, concatenate with "；".
        - Embedding: weighted average by importance.
        - ID, created_at from the base (higher-importance) item.
        """
        base, other = (b, a) if b.importance > a.importance else (a, b)

        merged = MemoryItem(
            id=base.id,
            content=base.content,
            importance=max(base.importance, other.importance),
            embedding=base.embedding,
            created_at=base.created_at,
            last_accessed=datetime.now(),
        )

        # Content: non-substring → concatenate
        if (other.content not in base.content
                and base.content not in other.content):
            merged.content = base.content + "；" + other.content
        elif len(other.content) > len(base.content):
            merged.content = other.content

        # Embedding: weighted average by importance
        if (base.embedding is not None and other.embedding is not None
                and len(base.embedding) == len(other.embedding)):
            w_a, w_b = base.importance, other.importance
            total = w_a + w_b
            if total > 0:
                merged.embedding = (base.embedding * w_a + other.embedding * w_b) / total

        return merged

    # ------------------------------------------------------------------
    # Persistence (PostgreSQL)
    # ------------------------------------------------------------------

    def _item_to_pg_row(self, item: MemoryItem) -> dict:
        return {
            "content": item.content,
            "importance": item.importance,
            "embedding": item.embedding.tolist() if item.embedding is not None else None,
            "category": item.category or "general",
            "tags": item.tags or [],
            "slot_hint": item.slot_hint or "",
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "last_accessed": item.last_accessed.isoformat() if item.last_accessed else None,
        }

    def _pg_item_id(self, item: MemoryItem) -> Optional[int]:
        """Return the PG-assigned id stored in item.id post-sync."""
        return item.id if hasattr(item, "id") and item.id >= 0 else None

    def _load_from_pg(self) -> None:
        """Load memory items from PostgreSQL (replaces ``load()``)."""
        if not self._username:
            return
        try:
            from medrag.infrastructure.storage.postgres_client import ltm_load_all

            rows = ltm_load_all(self._username)
            self._items.clear()
            self._vocab_id.clear()
            self._vocab.clear()
            self._content_hashes.clear()
            for r in rows:
                embedding = (
                    np.array(r["embedding"], dtype=np.float64)
                    if r.get("embedding") else None
                )
                created = r.get("created_at")
                accessed = r.get("last_accessed")
                item = MemoryItem(
                    id=r["id"],
                    content=r["content"],
                    importance=r["importance"] or 0.0,
                    embedding=embedding,
                    category=r.get("category", "general"),
                    tags=r.get("tags") or [],
                    slot_hint=r.get("slot_hint", ""),
                    created_at=created if isinstance(created, datetime) else None,
                    last_accessed=accessed if isinstance(accessed, datetime) else None,
                )
                self._items.append(item)
                self._build_vocab(item.content)
                self._content_hashes.add(self._hash(item.content))
            self._next_id = (max(item.id for item in self._items) + 1) if self._items else 0
        except Exception:
            logger.debug("Failed to load LTM from PG for user %s", self._username, exc_info=True)

    def _auto_save(self) -> None:
        """After every store operation, sync the latest item to PostgreSQL."""
        if not self._username or not self._items:
            return
        try:
            from medrag.infrastructure.storage.postgres_client import ltm_save_item

            item = self._items[-1]
            row = self._item_to_pg_row(item)
            pg_id = ltm_save_item(
                username=self._username,
                content=row["content"],
                importance=row["importance"],
                embedding=row["embedding"],
                category=row["category"],
                tags=row["tags"],
                slot_hint=row["slot_hint"],
                created_at=row["created_at"],
            )
            # Sync PG-assigned id back to in-memory item
            item.id = pg_id
            if pg_id >= self._next_id:
                self._next_id = pg_id + 1
        except Exception:
            logger.debug("LTM auto-save failed for user %s", self._username, exc_info=True)

    def _sync_consolidation_to_pg(self, result: ConsolidationResult) -> None:
        """After consolidation, sync deletions and updates to PostgreSQL."""
        if not self._username:
            return
        try:
            from medrag.infrastructure.storage.postgres_client import (
                ltm_delete_items,
                ltm_update_item,
            )

            if result.deleted_ids:
                ltm_delete_ids = [i for i in result.deleted_ids if i >= 0]
                if ltm_delete_ids:
                    ltm_delete_items(ltm_delete_ids)
            for item in result.updated_items:
                row = self._item_to_pg_row(item)
                ltm_update_item(
                    item_id=item.id,
                    content=row["content"],
                    importance=row["importance"],
                    embedding=row["embedding"],
                    tags=row["tags"],
                    last_accessed=row["last_accessed"],
                )
        except Exception:
            logger.debug("Failed to sync consolidation to PG", exc_info=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[MemoryItem]:
        return self._items

    def sync_last_item_id(self, pg_id: int) -> None:
        """Sync last item's ID with persistence-assigned ID.

        Mirrors AGI-saber LongTerm.SyncLastItemPGID().
        """
        if self._items and pg_id > 0:
            self._items[-1].id = pg_id
            if pg_id >= self._next_id:
                self._next_id = pg_id + 1
