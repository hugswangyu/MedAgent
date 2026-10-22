"""Schema-driven slots with per-slot and global budgets; required slots never vanish."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


def estimate_tokens(text: str) -> int:
    """Conservative UTF-8 byte budget for byte-based tokenizers, not an exact count.

    Callers may inject their model's tokenizer through token_counter. Message framing
    and output reserves must be budgeted separately by the runtime.
    """
    return len(text.encode("utf-8"))


PRIORITY_MEMORY = 100
PRIORITY_CASE_SUMMARY = 90
PRIORITY_KG = 80
PRIORITY_QA = 70
PRIORITY_CASE_CHUNKS = 60
PRIORITY_QUERY = 110


class ContextText(str):
    """A normal string carrying slot metadata through the existing service API."""
    def __new__(cls, value, slots=()):
        instance = super().__new__(cls, value)
        instance.slots = tuple(slots)
        return instance


class ContextBudgetError(ValueError):
    """Required instructions/query cannot fit: fail explicitly instead of dropping them."""


@dataclass(frozen=True)
class SlotSpec:
    priority: int
    token_budget: int
    required: bool = False


@dataclass(frozen=True)
class RuntimeContextSchema:
    slots: dict

    @classmethod
    def for_mode(cls, mode="chat"):
        common = {
            "constraints": SlotSpec(200, 4096, True),
            "profile": SlotSpec(100, 2048),
            "recall_memory": SlotSpec(60, 1024),
            "short_term": SlotSpec(40, 1024),
        }
        if mode in {"tool", "react"}:
            common["tool_state"] = SlotSpec(120, 4096, True)
        if mode == "react":
            common.update(planner=SlotSpec(110, 512, True), task_memory=SlotSpec(80, 2048))
        return cls(common)  # unknown mode safely uses chat schema


@dataclass
class Slot:
    kind: str
    content: Optional[str]
    priority: int = 50
    token_count: int = 0
    token_budget: Optional[int] = None
    required: bool = False


class ContextAssembler:
    def __init__(self, budget=4096, token_counter: Callable[[str], int] = estimate_tokens):
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        self._slots, self._budget = [], budget
        self._counter = token_counter
        self._rendered = ""

    def add(self, kind, content, priority=50, token_budget=None, required=False):
        if not content or not content.strip():
            if required:
                raise ContextBudgetError(f"Required slot {kind!r} is empty")
            return
        self._slots.append(Slot(kind, content.strip(), priority,
                                self._counter(content.strip()), token_budget, required))

    def add_schema(self, schema, kind, content):
        spec = schema.slots.get(kind)
        if spec:
            self.add(kind, content, spec.priority, spec.token_budget, spec.required)

    def _trim(self, text, budget):
        if self._counter(text) <= budget:
            return text
        suffix = "…"
        if self._counter(suffix) > budget:
            return ""
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo+hi+1)//2
            if self._counter(text[:mid]+suffix) <= budget:
                lo = mid
            else:
                hi = mid-1
        return text[:lo]+suffix

    def assemble(self):
        slots = sorted(self._slots, key=lambda s: s.priority, reverse=True)
        for slot in slots:
            if slot.content is None:
                continue
            if slot.token_budget is not None and self._counter(slot.content) > slot.token_budget:
                if slot.required:
                    raise ContextBudgetError(f"Required slot {slot.kind!r} exceeds its budget")
                slot.content = self._trim(slot.content, slot.token_budget) or None
            slot.token_count = self._counter(slot.content or "")
        render = lambda: "\n\n".join(s.content for s in slots if s.content)
        # Required content is reserved first, independent of numeric priority.
        required = "\n\n".join(s.content for s in slots if s.required and s.content)
        if self._counter(required) > self._budget:
            raise ContextBudgetError("Required context exceeds global budget")
        for slot in reversed(slots):
            if self._counter(render()) <= self._budget:
                break
            if not slot.required:
                slot.content = None
        self._rendered = render()
        self._slots = slots
        return ContextText(self._rendered, [s for s in slots if s.content])

    @property
    def total_tokens(self):
        return self._counter(self._rendered)

    @property
    def used_slots(self):
        return [s.kind for s in self._slots if s.content]

    @property
    def dropped_slots(self):
        return [s.kind for s in self._slots if s.content is None]

    def reset(self):
        self._slots.clear()
        self._rendered = ""
