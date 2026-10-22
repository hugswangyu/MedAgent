"""Health profile: an active-fact projection, not an overwriting category KV store."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from .extraction import FACT_EXTRACTION_PROMPT, extract_llm, extract_rules
from .types import AtomicFact

# Kept for callers importing the extraction prompt; contains no format placeholders.
_PREFERENCE_EXTRACT_PROMPT = FACT_EXTRACTION_PROMPT


class PreferenceStore:
    def __init__(self, username: str = "", long_term=None):
        if long_term is None:
            from .long_term import LongTermMemory
            long_term = LongTermMemory(username=username)
        self._username = username
        self._ltm = long_term
        self._legacy: Dict[str, str] = {}
        if username:
            self._load_from_pg()

    def _load_from_pg(self):
        """Old KV rows are read-only, explicitly unverified; never invent provenance."""
        try:
            from medrag.infrastructure.storage.postgres_client import pref_load_all
            self._legacy = pref_load_all(self._username)
        except Exception:
            self._legacy = {}

    def facts(self, predicate: Optional[str] = None, include_history: bool = False):
        facts = [x.fact for x in self._ltm.items if x.fact
                 and (predicate is None or x.fact.predicate == predicate)
                 and (include_history or x.fact.status == "active")]
        return sorted(facts, key=lambda f: (f.recorded_at, f.version))

    def current_facts(self):
        """Latest effective observation per vital; all historical samples stay stored."""
        from .extraction import TEMPORAL
        facts = self.facts()
        latest = {}
        for fact in facts:
            if fact.predicate in TEMPORAL:
                key = (fact.subject, fact.predicate)
                rank = (fact.effective_at or fact.recorded_at, fact.recorded_at)
                if key not in latest or rank > latest[key][0]:
                    latest[key] = (rank, fact)
        return [f for f in facts if f.predicate not in TEMPORAL
                or latest[(f.subject, f.predicate)][1] is f]

    def save(self, key: str, value: str):
        """Explicit/trusted API; conversation paths use grounded extractors."""
        if key and value:
            fact = AtomicFact(subject="user", predicate=key, value=value,
                              evidence=value, source_message_id=uuid4().hex)
            self._ltm.store_fact(fact)

    def save_batch(self, kvs):
        for key, value in kvs.items():
            self.save(key, value)

    def values(self, key):
        return list(dict.fromkeys(("否认：" if f.negated else "")+f.value
                                  for f in self.facts(key) if f.subject == "user"))

    def get(self, key, default=None):
        values = list(dict.fromkeys(("否认：" if f.negated else "")+f.value
                                    for f in self.current_facts() if f.predicate == key and f.subject == "user"))
        return "；".join(values) if values else self._legacy.get(key, default)

    def all(self):
        keys = dict.fromkeys(f.predicate for f in self.facts())
        return {key: self.get(key) for key in keys if self.get(key) is not None}

    @property
    def data(self):
        return self.all()

    @staticmethod
    def _has_llm():
        from medrag.llm.provider import is_llm_provider_configured
        return is_llm_provider_configured()

    def extract_sync(self, text, message_id=None, session_id="", recorded_at=None):
        facts = extract_rules(text, message_id or uuid4().hex, session_id, recorded_at)
        for fact in facts:
            self._ltm.store_fact(fact)
        return facts

    def llm_extract(self, text, message_id=None, session_id="", recorded_at=None):
        """Enrichment only. Synchronous rule facts have already been stored."""
        facts = extract_llm(text, message_id or uuid4().hex, session_id, recorded_at)
        for fact in facts:
            self._ltm.store_fact(fact)
        return bool(facts)

    def _regex_fallback(self, text):
        return bool(self.extract_sync(text))

    def build_context(self):
        facts = self.current_facts()
        if not facts and not self._legacy:
            return ""
        lines = [f.render() for f in facts]
        lines += [f"旧画像（来源与时间未知，未经确认）{k}: {v}" for k, v in self._legacy.items()
                  if not self.facts(k)]
        return "【用户健康画像/用户偏好】\n"+"\n".join(lines)
