"""Short-term memory — sliding window conversation context.

Mirrors AGI-saber internal/memory/memory.go ShortTerm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class ConversationMessage:
    """Single chat turn."""
    role: str       # "user" | "assistant" | "system"
    content: str
    timestamp: str = ""


class ShortTermMemory:
    """Sliding window over recent N turns of conversation.

    Each turn = one user message + one assistant response.
    ``max_turns`` controls how many complete turns are retained (``max_turns * 2`` entries).
    """

    def __init__(self, max_turns: int = 5):
        self._max_turns = max_turns
        self._messages: List[ConversationMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a message. Evicts oldest entries when window is full."""
        self._messages.append(ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().strftime("%H:%M:%S"),
        ))
        max_entries = self._max_turns * 2
        if len(self._messages) > max_entries:
            self._messages = self._messages[-max_entries:]

    def messages(self) -> List[ConversationMessage]:
        """Return a copy of current messages."""
        return list(self._messages)

    def to_llm_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI-style message list (for injection into chat history)."""
        return [
            {"role": m.role, "content": m.content}
            for m in self._messages
            if m.role in ("user", "assistant")
        ]

    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def __len__(self) -> int:
        return len(self._messages)
