"""Memory system shared data types.

Mirrors AGI-saber internal/memory/memory.go Item, RecallFilter,
ConsolidationConfig, ConsolidationResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MemoryItem:
    """Single long-term memory entry.

    Attributes mirroring AGI-saber memory.Item:
      id, content, importance (0-1), embedding, score (recall, not persisted),
      created_at, last_accessed, category, tags, slot_hint.
    """

    id: int
    content: str
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None
    score: float = 0.0

    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None

    # Schema-driven assembly fields (used by runtime slot filtering)
    category: str = "general"   # identity | preference | fact | episodic | tool_failure | policy | general
    tags: List[str] = field(default_factory=list)
    slot_hint: str = ""          # profile | recall | constraints | tool_state

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "importance": self.importance,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "score": self.score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "category": self.category,
            "tags": list(self.tags),
            "slot_hint": self.slot_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> MemoryItem:
        emb = None
        if data.get("embedding") is not None:
            emb = np.array(data["embedding"], dtype=np.float64)
        created = None
        if data.get("created_at"):
            created = datetime.fromisoformat(data["created_at"])
        last_acc = None
        if data.get("last_accessed"):
            last_acc = datetime.fromisoformat(data["last_accessed"])
        return cls(
            id=data["id"],
            content=data["content"],
            importance=data.get("importance", 0.5),
            embedding=emb,
            score=data.get("score", 0.0),
            created_at=created,
            last_accessed=last_acc,
            category=data.get("category", "general"),
            tags=list(data.get("tags", [])),
            slot_hint=data.get("slot_hint", ""),
        )


@dataclass
class RecallFilter:
    """Filter constraints for semantic recall.

    Mirrors AGI-saber memory.RecallFilter.
    - categories: only return items whose category matches one of these.
    - require_tags: items must contain all these tags.
    - min_score: relevance score threshold (default 0.4).
    - top_k: max results (0 = no limit).
    - max_age_hours: items older than this are filtered out (0 = no limit).
    """

    categories: Optional[List[str]] = None
    require_tags: Optional[List[str]] = None
    min_score: float = 0.4
    top_k: int = 5
    max_age_hours: Optional[float] = None


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation.

    Mirrors AGI-saber memory.ConsolidationConfig.
    """

    similarity_threshold: float = 0.80   # merge items above this (below dedup threshold)
    dedup_threshold: float = 0.95        # treat as duplicate above this
    ttl_days: int = 30                   # days before expiry (0 = never)
    decay_rate: float = 0.995            # daily importance decay multiplier
    min_importance: float = 0.3          # items below this + over TTL are expired
    trigger_interval: int = 5            # run consolidation every N stores


@dataclass
class ConsolidationResult:
    """Result of a consolidation cycle.

    Mirrors AGI-saber memory.ConsolidationResult.
    """

    deduped: int = 0
    merged: int = 0
    expired: int = 0
    deleted_ids: List[int] = field(default_factory=list)
    updated_items: List[MemoryItem] = field(default_factory=list)
