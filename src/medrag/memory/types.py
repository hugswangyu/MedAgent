"""Memory system shared data types.

Mirrors AGI-saber internal/memory/memory.go Item, RecallFilter,
ConsolidationConfig, ConsolidationResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from uuid import uuid4
from datetime import timezone


def _datetime(value):
    """Normalize storage/API timestamps to naive UTC for legacy PG compatibility."""
    if value is None:
        return None
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if result.tzinfo:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


@dataclass
class AtomicFact:
    """One grounded proposition; self_reported is not a clinical confirmation."""

    subject: str
    predicate: str
    value: str
    evidence: str
    source_message_id: str
    source_session_id: str = ""
    source_role: str = "user"
    recorded_at: datetime = field(default_factory=datetime.utcnow)
    effective_at: Optional[datetime] = None  # unknown unless explicitly stated
    confirmation_status: str = "self_reported"
    negated: bool = False
    operation: str = "add"  # add | update | retract
    replaces_value: str = ""
    fact_id: str = field(default_factory=lambda: uuid4().hex)
    version: int = 1
    supersedes: List[str] = field(default_factory=list)
    status: str = "active"  # active | superseded | retracted

    @property
    def key(self):
        return (self.subject, self.predicate, self.value, self.negated)

    def render(self) -> str:
        polarity = "否认：" if self.negated else ""
        effective = f"，生效于{self.effective_at.isoformat()}" if self.effective_at else ""
        return (f"{self.subject} · {self.predicate}: {polarity}{self.value}"
                f"（{self.confirmation_status}，记录于{self.recorded_at.isoformat()}{effective}）")

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        data = asdict(self)
        data["recorded_at"] = self.recorded_at.isoformat()
        data["effective_at"] = self.effective_at.isoformat() if self.effective_at else None
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> AtomicFact:
        from dataclasses import fields
        values = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        values["recorded_at"] = _datetime(data.get("recorded_at")) or datetime.utcnow()
        values["effective_at"] = _datetime(data.get("effective_at"))
        return cls(**values)



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

    # Decay always derives from this baseline, never from an already decayed score.
    base_importance: Optional[float] = None
    decay_anchor: Optional[datetime] = None
    fact: Optional[AtomicFact] = None

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
            "base_importance": self.base_importance,
            "decay_anchor": self.decay_anchor.isoformat() if self.decay_anchor else None,
            "fact": self.fact.to_dict() if self.fact else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> MemoryItem:
        emb = None
        if data.get("embedding") is not None:
            emb = np.array(data["embedding"], dtype=np.float64)
        created = None
        if data.get("created_at"):
            created = _datetime(data["created_at"])
        last_acc = None
        if data.get("last_accessed"):
            last_acc = _datetime(data["last_accessed"])
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
            base_importance=data.get("base_importance", data.get("importance", 0.5)),
            decay_anchor=(_datetime(data.get("decay_anchor")) or created)
                if data.get("base_importance") is not None else datetime.utcnow(),
            fact=AtomicFact.from_dict(data["fact"]) if data.get("fact") else None,
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
