"""User-scoped atomic facts, semantic recall and idempotent lifecycle maintenance."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import List, Optional

import numpy as np

from .extraction import SINGLETON, TEMPORAL
from .types import AtomicFact, ConsolidationConfig, ConsolidationResult, MemoryItem, RecallFilter

logger = logging.getLogger(__name__)


def _cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _tokenize(text):
    return re.findall(r"[一-鿿]|[a-z0-9]+", text.lower())


def _text_similarity(a, b):
    vocab = sorted(set(_tokenize(a) + _tokenize(b)))
    ta, tb = _tokenize(a), _tokenize(b)
    return _cosine(np.array([ta.count(t) for t in vocab]), np.array([tb.count(t) for t in vocab]))


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class LongTermMemory:
    def __init__(self, username: str = "", persist_path: Optional[str] = None):
        self._username, self._persist_path = username, persist_path
        self._lock = RLock()
        self._items: List[MemoryItem] = []
        self._next_id = self._store_count = 0
        self._consolidation_cfg = None
        self._content_hashes = set()
        self._content_index = {}
        self._pg_ids = set()
        self._dirty = []
        self._deleted = set()
        if persist_path:
            self._load_from_json()
        else:
            self._load_from_pg()

    def set_consolidation_config(self, cfg):
        self._consolidation_cfg = cfg

    @property
    def consolidation_cfg(self):
        return self._consolidation_cfg

    @staticmethod
    def _hash(content):
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def _rebuild_hashes(self):
        self._content_hashes = {self._hash(x.content) for x in self._items}
        self._content_index = {self._hash(x.content): x for x in self._items if not x.fact}

    def _new_item(self, content, importance, embedding, category, tags, slot_hint, fact=None):
        now = datetime.utcnow()
        item = MemoryItem(id=self._next_id, content=content, importance=importance,
                          base_importance=importance, embedding=embedding, created_at=now,
                          decay_anchor=now, last_accessed=now, category=category,
                          tags=list(tags or []), slot_hint=slot_hint, fact=fact)
        self._next_id += 1
        self._store_count += 1
        self._items.append(item)
        self._content_hashes.add(self._hash(content))
        if not fact:
            self._content_index[self._hash(content)] = item
        return item

    @locked
    def store_fact(self, fact: AtomicFact, importance: float = 0.8,
                   embedding: Optional[np.ndarray] = None) -> bool:
        """Keep provenance/history; distinct structured propositions never cosine-merge."""
        if fact.source_role != "user" or fact.confirmation_status not in {"self_reported", "confirmed"}:
            return False
        active = [x for x in self._items if x.fact and x.fact.status == "active"
                  and (x.fact.subject, x.fact.predicate) == (fact.subject, fact.predicate)]
        # Async results from an earlier message cannot overwrite a newer correction.
        history = [x.fact for x in self._items if x.fact and
                   (x.fact.subject, x.fact.predicate) == (fact.subject, fact.predicate)]
        same_message = [x for x in self._items if x.fact and x.fact.key == fact.key
                        and x.fact.source_message_id == fact.source_message_id]
        if same_message:
            return False
        for existing in active:
            old = existing.fact
            # New observations of a vital sign keep their own timestamp/provenance.
            if old.key == fact.key and fact.predicate not in TEMPORAL and fact.operation == "add":
                existing.last_accessed = datetime.utcnow()
                self._persist([existing])
                return False
        targets = [x for x in active if (
            (fact.predicate in SINGLETON)
            or (fact.replaces_value and x.fact.value == fact.replaces_value and not x.fact.negated)
            or (fact.value == x.fact.value and fact.negated != x.fact.negated)
            or (fact.operation in {"update", "retract"} and fact.value == x.fact.value)
        )]
        if fact.predicate in TEMPORAL and targets:
            targets = [max(targets, key=lambda item: item.fact.recorded_at)]
        changed = []
        # Delayed extraction is retained as historical, never resurrected into the profile.
        newer = [f for f in history if f.recorded_at > fact.recorded_at and (
            fact.predicate in SINGLETON or f.value == fact.value or f.replaces_value == fact.value)]
        if newer:
            fact.status = "superseded"
        else:
            for old in targets:
                if old.fact.recorded_at <= fact.recorded_at:
                    old.fact.status = "superseded"
                    fact.supersedes.append(old.fact.fact_id)
                    changed.append(old)
            if fact.operation == "retract":
                fact.status = "retracted"
        fact.version = 1 + max((f.version for f in history), default=0)
        category = "identity" if fact.predicate in {"姓名", "城市", "职业"} else "fact"
        item = self._new_item(f"{fact.subject} {fact.predicate}: {'否认 ' if fact.negated else ''}{fact.value}",
                              importance, embedding, category, ["atomic_fact"], "profile", fact)
        self._persist(changed + [item])
        return True

    def store(self, content, importance=0.5, embedding=None):
        return self.store_classified(content, importance, embedding)

    @locked
    def store_classified(self, content, importance=0.5, embedding=None,
                         category="general", tags=None, slot_hint=""):
        """Trusted/manual compatibility API. Conversation ingestion uses store_fact."""
        if not content.strip():
            return False
        existing = self._content_index.get(self._hash(content))
        if existing is None and embedding is not None and self._consolidation_cfg is not None:
            for candidate in self._items:
                if candidate.fact or candidate.embedding is None or len(embedding) != len(candidate.embedding):
                    continue
                if _cosine(embedding, candidate.embedding) >= self._consolidation_cfg.dedup_threshold:
                    existing = candidate
                    break
        if existing is not None:
            existing.importance = max(existing.importance, importance)
            existing.base_importance = max(existing.base_importance or 0, importance)
            existing.last_accessed = datetime.utcnow()
            existing.tags = list(dict.fromkeys(existing.tags + (tags or [])))
            if existing.category == "general":
                existing.category = category
            existing.slot_hint = existing.slot_hint or slot_hint
            self._persist([existing])
            return False
        item = self._new_item(content, importance, embedding, category or "general", tags, slot_hint)
        self._persist([item])
        return True

    @locked
    def store_item(self, item):
        item.created_at = item.created_at or datetime.utcnow()
        item.last_accessed = item.last_accessed or item.created_at
        item.base_importance = item.base_importance if item.base_importance is not None else item.importance
        item.decay_anchor = item.decay_anchor or item.created_at
        self._items.append(item)
        self._next_id = max(self._next_id, item.id + 1)
        self._rebuild_hashes()
        self._persist([item])

    def recall(self, query="", top_k=5, query_embedding=None):
        return self.recall_by_filter(query, query_embedding, RecallFilter(top_k=top_k))

    @locked
    def recall_by_filter(self, query="", query_embedding=None, filter=None):
        f, now = filter or RecallFilter(), datetime.utcnow()
        scored = []
        for item in self._items:
            if item.fact and (item.fact.status != "active" or item.fact.confirmation_status == "unconfirmed"):
                continue
            if f.categories and item.category not in f.categories:
                continue
            if f.require_tags and not all(t in item.tags for t in f.require_tags):
                continue
            if f.max_age_hours and item.created_at and (now-item.created_at).total_seconds()/3600 > f.max_age_hours:
                continue
            sim = self._compute_similarity(item, query, query_embedding)
            score = sim * .7 + item.importance * .3
            if score >= f.min_score:
                item.score = score
                scored.append(item)
        scored.sort(key=lambda x: x.score, reverse=True)
        selected = scored[:f.top_k] if f.top_k > 0 else scored
        for item in selected:
            item.last_accessed = now
        if selected:
            self._persist(selected)
        return selected

    def _compute_similarity(self, item, query, query_emb):
        if query_emb is not None and item.embedding is not None and len(query_emb) == len(item.embedding):
            return _cosine(query_emb, item.embedding)
        return _text_similarity(query, item.content)

    def filter_by_category(self, categories, limit=0):
        items = [x for x in self.items if x.category in categories and (not x.fact or x.fact.status == "active")]
        return items[:limit] if limit > 0 else items

    def need_consolidation(self):
        cfg = self._consolidation_cfg
        return bool(cfg and cfg.trigger_interval > 0 and self._store_count >= cfg.trigger_interval)

    @locked
    def consolidate(self, now=None):
        result, cfg = ConsolidationResult(), self._consolidation_cfg
        if cfg is None:
            return result
        now = now or datetime.utcnow()
        self._store_count = 0
        changed, removed = [], set()
        for item in self._items:
            if item.base_importance is None:
                item.base_importance = item.importance
            anchor = item.decay_anchor or item.created_at or now
            item.decay_anchor = anchor
            new_imp = item.base_importance * math.pow(cfg.decay_rate, max(0., (now-anchor).total_seconds()/86400))
            if new_imp != item.importance:
                item.importance = new_imp
                changed.append(item)
        # Never merge distinct atomic facts: polarity, time and versions are meaningful.
        for i, a in enumerate(self._items):
            if a.id in removed or a.fact:
                continue
            for b in self._items[i+1:]:
                if b.id in removed or b.fact:
                    continue
                sim = self._item_similarity(a, b)
                if sim >= cfg.dedup_threshold:
                    keep, discard = (a, b) if a.importance >= b.importance else (b, a)
                    removed.add(discard.id)
                    keep.tags = list(dict.fromkeys(a.tags+b.tags))
                    changed.append(keep)
                    result.deduped += 1
                    if discard is a:
                        break
                elif sim >= cfg.similarity_threshold:
                    merged = self._merge_items(a, b)
                    # Preserve stable ID; never delete the same ID used by the merged row.
                    merged.id = a.id
                    self._items[i] = a = merged
                    removed.add(b.id)
                    changed.append(merged)
                    result.merged += 1
        for item in self._items:
            if item.id in removed:
                continue
            age = (now-(item.created_at or now)).total_seconds()/86400
            # Revision history is retained for audit; active facts use the normal TTL gate.
            if item.fact and item.fact.status != "active":
                continue
            if cfg.ttl_days > 0 and age > cfg.ttl_days and item.importance < cfg.min_importance:
                removed.add(item.id)
                result.expired += 1
        self._items = [x for x in self._items if x.id not in removed]
        self._rebuild_hashes()
        result.deleted_ids = sorted(removed)
        result.updated_items = list({x.id: x for x in changed if x.id not in removed}.values())
        self._persist(result.updated_items, result.deleted_ids)
        return result

    def _item_similarity(self, a, b):
        return self._compute_similarity(a, b.content, b.embedding)

    @staticmethod
    def _merge_items(a, b):
        from copy import deepcopy
        base, other = (a, b) if a.importance >= b.importance else (b, a)
        merged = deepcopy(base)
        merged.importance = max(a.importance, b.importance)
        merged.base_importance = merged.importance
        merged.decay_anchor = datetime.utcnow()
        merged.tags = list(dict.fromkeys(a.tags+b.tags))
        merged.slot_hint = base.slot_hint or other.slot_hint
        merged.category = other.category if base.category == "general" else base.category
        if other.content not in base.content:
            merged.content = other.content if base.content in other.content else base.content+"；"+other.content
        if a.embedding is not None and b.embedding is not None and len(a.embedding) == len(b.embedding):
            total = a.importance+b.importance
            if total:
                merged.embedding = (a.embedding*a.importance+b.embedding*b.importance)/total
        merged.last_accessed = datetime.utcnow()
        return merged

    def _load_from_json(self):
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("items", []) if isinstance(data, dict) else data
            self._restore(rows)
        except Exception:
            logger.exception("Failed to load memory JSON")

    def _restore(self, rows):
        self._items = [MemoryItem.from_dict(row) for row in rows]
        self._next_id = max((x.id for x in self._items), default=-1)+1
        self._rebuild_hashes()

    def _load_from_pg(self):
        if not self._username:
            return
        try:
            from medrag.infrastructure.storage.postgres_client import ltm_load_all
            self._restore(ltm_load_all(self._username))
            self._pg_ids = {x.id for x in self._items}
        except Exception:
            logger.warning("LTM load unavailable", exc_info=True)

    def _persist(self, changed=(), deleted=()):
        if self._persist_path:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name+".tmp")
            tmp.write_text(json.dumps({"items": [x.to_dict() for x in self._items]}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
            return
        if not self._username:
            return
        self._deleted.update(deleted)
        for item in changed:
            self._dirty = [x for x in self._dirty if x.id != item.id]
            self._dirty.append(item)
        self._dirty = [x for x in self._dirty if x.id not in self._deleted]
        try:
            from medrag.infrastructure.storage.postgres_client import ltm_sync_items
            rows = [dict(x.to_dict(), id=x.id if x.id in self._pg_ids else None) for x in self._dirty]
            ids = ltm_sync_items(self._username, rows, sorted(self._deleted & self._pg_ids))
            for item, pg_id in zip(self._dirty, ids):
                item.id = pg_id
                self._pg_ids.add(pg_id)
                self._next_id = max(self._next_id, pg_id+1)
            self._pg_ids.difference_update(self._deleted)
            self._dirty.clear()
            self._deleted.clear()
        except Exception:
            logger.warning("LTM sync failed; pending changes retained for retry", exc_info=True)

    @locked
    def clear(self):
        ids = [x.id for x in self._items]
        self._items.clear()
        self._content_hashes.clear()
        self._content_index.clear()
        self._store_count = 0
        self._persist(deleted=ids)

    def count(self):
        return len(self._items)

    @property
    def items(self):
        with self._lock:
            return list(self._items)

    def sync_last_item_id(self, pg_id):
        with self._lock:
            if self._items and pg_id > 0:
                self._items[-1].id = pg_id
                self._next_id = max(self._next_id, pg_id+1)
