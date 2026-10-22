"""Three memory layers: recent dialogue, grounded long-term facts, health profile."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from threading import RLock, BoundedSemaphore
from uuid import uuid4

from .long_term import LongTermMemory
from .preference import PreferenceStore
from .schema import (
    ContextAssembler, Slot, estimate_tokens,
    PRIORITY_MEMORY, PRIORITY_CASE_SUMMARY, PRIORITY_KG,
    PRIORITY_QA, PRIORITY_CASE_CHUNKS, PRIORITY_QUERY,
)
from .short_term import ConversationMessage, ShortTermMemory
from .types import AtomicFact, ConsolidationConfig, ConsolidationResult, MemoryItem, RecallFilter

logger = logging.getLogger(__name__)


class MemorySystem:
    def __init__(self, max_turns=5, consolidation=None, persist_path=None, username="",
                 session_id="", embedder=None, async_extraction=True, long_term=None):
        self._persist_path, self._username = persist_path, username
        self._session_id, self._embedder = session_id, embedder
        self._lock = RLock()
        self._generation = 0
        self._msg_count = 0
        self._async_extraction = async_extraction
        self._executor = None
        self._pending = set()
        self._capacity = BoundedSemaphore(16)
        self.short_term = ShortTermMemory(max_turns=max_turns)
        self.long_term = long_term if long_term is not None else LongTermMemory(username, persist_path)
        self._consolidation_cfg = consolidation or ConsolidationConfig()
        self.long_term.set_consolidation_config(self._consolidation_cfg)
        self.preferences = PreferenceStore(username=username, long_term=self.long_term)

    def _store_fact(self, fact):
        embedding = None
        if self._embedder:
            try:
                embedding = self._embedder(f"{fact.subject} {fact.predicate}: {'否认 ' if fact.negated else ''}{fact.value}")
            except Exception:
                logger.warning("Fact embedding unavailable; using text recall")
        self.long_term.store_fact(fact, embedding=embedding)

    def add_message(self, role, content):
        with self._lock:
            self.short_term.add(role, content)
            self._msg_count += 1
            if role != "user":
                return
            message_id, recorded_at = uuid4().hex, datetime.utcnow()
            generation = self._generation
            # Rules execute before returning, independently of LLM availability.
            from .extraction import extract_rules
            for fact in extract_rules(content, message_id, self._session_id, recorded_at):
                self._store_fact(fact)
            if not self._async_extraction or not self._capacity.acquire(blocking=False):
                return
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-extract")
            future = self._executor.submit(self._enrich, content, message_id, recorded_at, generation)
            self._pending.add(future)
            future.add_done_callback(self._extraction_done)

    def _enrich(self, text, message_id, recorded_at, generation):
        from .extraction import extract_llm
        facts = extract_llm(text, message_id, self._session_id, recorded_at)
        with self._lock:
            if generation != self._generation:
                return
            for fact in facts:
                self._store_fact(fact)
            self._maybe_consolidate()

    def _extraction_done(self, future):
        with self._lock:
            self._pending.discard(future)
        self._capacity.release()
        if not future.cancelled() and future.exception():
            logger.error("Memory enrichment failed: %s", future.exception())

    def wait_for_extraction(self, timeout=20):
        with self._lock:
            pending = list(self._pending)
        return not wait(pending, timeout=timeout).not_done

    def close(self):
        with self._lock:
            self._generation += 1
            executor, self._executor = self._executor, None
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)

    def add_message_with_embedding(self, role, content, embedding):
        # Whole-message embeddings must never be assigned to extracted atomic facts.
        self.add_message(role, content)

    def store_assistant_reply(self, content, embedding=None):
        with self._lock:
            self.short_term.add("assistant", content)
            self._maybe_consolidate()

    def remember(self, content, importance=.5, embedding=None, category="general", tags=None, slot_hint=""):
        """Explicit/trusted insertion API; never called on raw chat replies."""
        return self.long_term.store_classified(content, importance, embedding, category, tags, slot_hint)

    def recall(self, query="", query_embedding=None, top_k=5, filter=None):
        return self.long_term.recall_by_filter(query, query_embedding,
                                               filter or RecallFilter(top_k=top_k))

    def build_context(self, query, query_embedding=None, include_stm=True, budget=4096):
        with self._lock:
            assembler = ContextAssembler(budget=budget)
            # Each fact is independently budgeted; never drop an entire profile at once.
            facts = self.preferences.current_facts()
            for fact in facts:
                critical = fact.predicate in {"过敏史", "慢性病史", "长期用药"}
                assembler.add("profile", "【用户健康画像/用户偏好】\n"+fact.render(),
                              priority=100 if critical else 90, token_budget=768,
                              required=critical)
            if not facts:
                assembler.add("profile", self.preferences.build_context(), priority=90, token_budget=1024)
            for item in self.recall(query, query_embedding):
                if item.fact and any(f.fact_id == item.fact.fact_id for f in facts):
                    continue
                text = item.fact.render() if item.fact else "来源未经核验的历史资料："+item.content
                assembler.add("recall_memory", "【长期记忆】\n"+text, priority=60, token_budget=512)
            if include_stm:
                for msg in reversed(self.short_term.to_llm_messages()[-6:]):
                    assembler.add("short_term", "【对话历史】\n"+msg["role"]+": "+msg["content"],
                                  priority=40, token_budget=512)
            return assembler.assemble()

    def _maybe_consolidate(self):
        if self.long_term.need_consolidation():
            self.long_term.consolidate()

    def consolidate_now(self):
        return self.long_term.consolidate()

    def clear(self):
        with self._lock:
            self._generation += 1
            self.short_term.clear()
            self.long_term.clear()
            self.preferences._legacy.clear()
            if self._username:
                from medrag.infrastructure.storage.postgres_client import pref_clear
                pref_clear(self._username)
            self._msg_count = 0

    @property
    def stm_messages(self):
        return self.short_term.messages()

    @property
    def stats(self):
        return {"stm_count": len(self.short_term), "ltm_count": self.long_term.count(),
                "preferences": len(self.preferences.data), "msg_count": self._msg_count}


_system = None


def get_memory_system():
    global _system
    if _system is None:
        _system = MemorySystem()
    return _system


def create_memory_system(max_turns=5, persist_path=None, username="", **kwargs):
    return MemorySystem(max_turns=max_turns, persist_path=persist_path, username=username, **kwargs)


__all__ = ["MemorySystem", "get_memory_system", "create_memory_system", "ShortTermMemory",
           "LongTermMemory", "PreferenceStore", "ContextAssembler", "Slot", "estimate_tokens",
           "AtomicFact", "MemoryItem", "RecallFilter", "ConsolidationConfig", "ConsolidationResult"]
