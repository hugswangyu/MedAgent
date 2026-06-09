"""User preference store — key-value pairs extracted from conversation.

Mirrors AGI-saber internal/memory/memory.go Preference.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


class PreferenceStore:
    """User preference key-value store with rule-based extraction.

    Supports LLM-based extraction as an upgrade path — for now,
    uses regex rules to capture common preference patterns from user messages.
    """

    def __init__(self):
        self._data: Dict[str, str] = {}

        # Rule patterns: (regex, key_name) — mirrors AGI-saber ExtractAndSave
        self._rules: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"我(?:叫|是|的名字(?:是|为)?)\s*(.+?)(?:[，。\.]|$)"), "姓名"),
            (re.compile(r"我(?:喜欢|爱)\s*(.+?)(?:[，。\.]|$)"), "喜好"),
            (re.compile(r"我(?:住在|来自|在)\s*(.+?)(?:[，。\.]|$)"), "城市"),
            (re.compile(r"我(?:今年|年龄)\s*(\d+)\s*岁"), "年龄"),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, key: str, value: str) -> None:
        if key and value:
            self._data[key] = value

    def save_batch(self, kvs: Dict[str, str]) -> None:
        for k, v in kvs.items():
            if k and v:
                self._data[k] = v

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(key, default)

    def all(self) -> Dict[str, str]:
        return dict(self._data)

    def extract_and_save(self, text: str) -> Optional[Tuple[str, str]]:
        """Try to extract a preference from *text* using rules.

        Returns (key, value) if matched, else None.
        Mirrors AGI-saber Preference.ExtractAndSave.
        """
        for pattern, key in self._rules:
            m = pattern.search(text)
            if m:
                value = m.group(1).strip()
                self.save(key, value)
                return (key, value)
        return None

    def build_context(self) -> str:
        """Format stored preferences as a context string for LLM prompts.

        Mirrors AGI-saber Preference.BuildContext.
        Returns empty string if no preferences stored.
        """
        if not self._data:
            return ""
        lines = [f"{k}: {v}" for k, v in self._data.items()]
        return "【用户偏好】\n" + "\n".join(lines)

    @property
    def data(self) -> Dict[str, str]:
        return self._data
