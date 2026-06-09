"""Memory system — unified facade for STM, LTM, preference, and graph memory.

Usage::

    from medrag.memory import MemorySystem, get_memory_system

    ms = MemorySystem()
    ms.add_message("user", "你好，我叫张三")
    ms.remember("患者有高血压病史", importance=0.8)

    context = ms.build_context("你好")
    # → "【用户偏好】\\n姓名: 张三\\n【长期记忆】\\n- 患者有高血压病史"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .classifier import classify_memory_content
from .graph_memory import GraphMemory
from .long_term import LongTermMemory
from .preference import PreferenceStore
from .short_term import ConversationMessage, ShortTermMemory
from .types import ConsolidationConfig, ConsolidationResult, MemoryItem, RecallFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemorySystem
# ---------------------------------------------------------------------------


class MemorySystem:
    """Unified memory facade integrating STM + GraphMemory + Preference.

    Mirrors the integration pattern from AGI-saber agent/agent.go process().
    """

    def __init__(
        self,
        max_turns: int = 5,
        consolidation: Optional[ConsolidationConfig] = None,
        kg_store=None,
    ):
        self.short_term = ShortTermMemory(max_turns=max_turns)
        self.preferences = PreferenceStore()
        self.long_term = LongTermMemory()
        self._consolidation_cfg = consolidation or ConsolidationConfig()
        self.long_term.set_consolidation_config(self._consolidation_cfg)
        self.graph = GraphMemory(self.long_term, kg_store=kg_store)
        self._msg_count = 0

    # ------------------------------------------------------------------
    # Message-level API (called per chat turn)
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Process a conversation message through all memory layers.

        1. STM: store in sliding window.
        2. Preference: extract from user messages via rules.
        3. LTM: if content is classifiable, store as long-term memory.

        Mirrors AGI-saber process() stm.Add + extractAndSave + extractMemoryFromReply.
        """
        self.short_term.add(role, content)
        self._msg_count += 1

        if role == "user":
            # ── Rule-based preference extraction (sync) ──
            self.preferences.extract_and_save(content)

            # ── Store classifiable content as long-term memory ──
            cat, tags, hint = classify_memory_content(content)
            if cat != "general":
                self.remember(content, importance=0.6,
                              category=cat, tags=tags, slot_hint=hint)

    def add_message_with_embedding(self, role: str, content: str,
                                    embedding: np.ndarray) -> None:
        """Like add_message but with an embedding vector for better recall.

        Preferred when an embedding model is available (e.g. from the RAG pipeline).
        """
        self.short_term.add(role, content)
        self._msg_count += 1

        if role == "user":
            self.preferences.extract_and_save(content)
            cat, tags, hint = classify_memory_content(content)
            if cat != "general":
                self.remember(content, importance=0.6,
                              embedding=embedding,
                              category=cat, tags=tags, slot_hint=hint)

    def store_assistant_reply(self, content: str,
                               embedding: Optional[np.ndarray] = None) -> None:
        """Process assistant reply: store in STM, extract memory-worthy info.

        Mirrors AGI-saber process() stm.Add("assistant") + extractMemoryFromReply.
        """
        self.short_term.add("assistant", content)

        # Classify and store assistant reply content
        cat, tags, hint = classify_memory_content(content)
        if cat != "general":
            self.remember(content, importance=0.5,
                          embedding=embedding,
                          category=cat, tags=tags, slot_hint=hint)

        # Check consolidation trigger
        self._maybe_consolidate()

    # ------------------------------------------------------------------
    # Long-term memory API
    # ------------------------------------------------------------------

    def remember(self, content: str, importance: float = 0.5,
                 embedding: Optional[np.ndarray] = None,
                 category: str = "general",
                 tags: Optional[List[str]] = None,
                 slot_hint: str = "") -> bool:
        """Store a long-term memory.

        Returns True if new, False if deduped.
        """
        tags = tags or []
        added, _ = self.graph.store_classified(
            content, importance, embedding, category, tags, slot_hint,
        )
        return added

    def recall(self, query: str = "",
               query_embedding: Optional[np.ndarray] = None,
               top_k: int = 5,
               filter: Optional[RecallFilter] = None) -> List[MemoryItem]:
        """Recall relevant long-term memories."""
        return self.graph.recall_by_filter(query, query_embedding, filter)

    # ------------------------------------------------------------------
    # Context building for LLM prompts
    # ------------------------------------------------------------------

    def build_context(self, query: str,
                      query_embedding: Optional[np.ndarray] = None,
                      include_stm: bool = True) -> str:
        """Build a formatted memory context string for LLM system prompt injection.

        Sections:
          1. User preferences (from PreferenceStore.build_context)
          2. Long-term memory recall (from recall)
          3. Short-term memory (recent conversation history)

        Mirrors AGI-saber buildContextPrefix via runtime.ContextAssembler.
        For Phase 1, this is a simplified single-block version.
        """
        parts: List[str] = []

        # 1. Preferences
        pref_text = self.preferences.build_context()
        if pref_text:
            parts.append(pref_text)

        # 2. Long-term recall
        ltm_results = self.recall(query, query_embedding, top_k=5)
        if ltm_results:
            mem_lines = ["【长期记忆】"]
            for item in ltm_results:
                mem_lines.append(f"- {item.content}")
            parts.append("\n".join(mem_lines))

        # 3. Short-term history
        if include_stm and len(self.short_term) > 0:
            stm_msgs = self.short_term.to_llm_messages()
            if stm_msgs:
                stm_lines = ["【对话历史】"]
                for m in stm_msgs[-6:]:  # last 3 turns
                    prefix = "用户" if m["role"] == "user" else "助手"
                    text = m["content"][:200]
                    stm_lines.append(f"{prefix}: {text}")
                parts.append("\n".join(stm_lines))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def _maybe_consolidate(self) -> None:
        """Trigger consolidation if threshold reached.

        Mirrors AGI-saber process() async consolidation goroutine.
        """
        if self.graph.need_consolidation():
            result = self.graph.graph_aware_consolidate()
            if result.deduped or result.merged or result.expired:
                logger.info(
                    "Memory consolidation: %d deduped, %d merged, %d expired",
                    result.deduped, result.merged, result.expired,
                )

    def consolidate_now(self) -> ConsolidationResult:
        """Force an immediate consolidation cycle."""
        result = self.graph.graph_aware_consolidate()
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all memory layers."""
        self.short_term.clear()
        self.long_term = LongTermMemory()
        self.graph = GraphMemory(self.long_term)
        self.preferences = PreferenceStore()
        self._msg_count = 0

    @property
    def stm_messages(self) -> List[ConversationMessage]:
        return self.short_term.messages()

    @property
    def stats(self) -> Dict:
        return {
            "stm_count": len(self.short_term),
            "ltm_count": self.long_term.count(),
            "preferences": len(self.preferences.data),
            "msg_count": self._msg_count,
        }


# ---------------------------------------------------------------------------
# Singleton (per-service-instance, reused across requests)
# ---------------------------------------------------------------------------

_system: Optional[MemorySystem] = None


def get_memory_system() -> MemorySystem:
    """Return the default singleton MemorySystem.

    Reused across requests within the same process.
    For per-session memory, create a MemorySystem per session_id.
    """
    global _system
    if _system is None:
        _system = MemorySystem()
    return _system


def create_memory_system(max_turns: int = 5,
                          kg_store=None) -> MemorySystem:
    """Create a fresh MemorySystem (useful for per-session isolation)."""
    return MemorySystem(max_turns=max_turns, kg_store=kg_store)


__all__ = [
    "MemorySystem",
    "get_memory_system",
    "create_memory_system",
    "ShortTermMemory",
    "LongTermMemory",
    "PreferenceStore",
    "GraphMemory",
    "MemoryItem",
    "RecallFilter",
    "ConsolidationConfig",
    "ConsolidationResult",
]
