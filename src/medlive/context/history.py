"""Disabled compatibility guard for the retired LiveRAG history feature."""

from __future__ import annotations

from typing import Any


class HistoryCompactor:
    """Reject all use of retired local long-term history persistence."""

    def __init__(self, **_: Any) -> None:
        pass

    async def compact_after_call(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError(
            "LiveRAG independent history is disabled; PostgreSQL is the sole fact source"
        )
