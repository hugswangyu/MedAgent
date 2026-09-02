"""会话持久化：PostgreSQL 存储，支持按用户隔离。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from medrag.infrastructure.storage.postgres_client import (
    session_list as _pg_session_list,
    session_get as _pg_session_get,
    session_delete as _pg_session_delete,
    message_list as _pg_message_list,
)
from medrag.infrastructure.storage import phase1_repository

from .schemas import SessionSummary, SessionMessage, SessionDetailResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公开 API（多一层封装，保持调用方不变）
# ---------------------------------------------------------------------------


def add_message(
    session_id: str,
    msg_type: str,
    content: str,
    rag_trace: Optional[dict] = None,
    username: str = "",
    user_id: str = "",
    turn_id: str = "",
) -> None:
    """Atomically append one compatibility and canonical text message."""

    if not username or not user_id or not turn_id:
        raise ValueError("username, user_id, and turn_id are required")
    if msg_type not in {"human", "ai"}:
        raise ValueError("unsupported text message type")
    phase1_repository.record_text_message(
        user_id=user_id,
        username=username,
        session_id=session_id,
        turn_id=turn_id,
        role="user" if msg_type == "human" else "assistant",
        content=content,
        rag_trace=rag_trace,
    )


def add_turn(
    session_id: str,
    *,
    user_text: str,
    assistant_text: str,
    username: str,
    user_id: str,
    turn_id: str,
    rag_trace: Optional[dict] = None,
) -> None:
    """Atomically persist both sides of one completed text turn."""

    phase1_repository.record_text_turn(
        user_id=user_id,
        username=username,
        session_id=session_id,
        turn_id=turn_id,
        user_text=user_text,
        assistant_text=assistant_text,
        rag_trace=rag_trace,
    )


def finalize_session(
    session_id: str, *, user_id: str, summary_version: int = 1
) -> dict:
    """Idempotently create the text summary and controlled memory candidates."""

    return phase1_repository.finalize_text_session_memory(
        user_id=user_id,
        session_id=session_id,
        summary_version=summary_version,
    )


def get_sessions(username: str) -> List[SessionSummary]:
    rows = _pg_session_list(username)
    result = []
    for r in rows:
        updated = r.get("updated_at", "")
        if hasattr(updated, "isoformat"):
            updated = updated.isoformat()
        result.append(
            SessionSummary(
                session_id=r["session_id"],
                message_count=r.get("message_count", 0),
                updated_at=str(updated),
            )
        )
    return result


def get_session(session_id: str, username: str) -> Optional[SessionDetailResponse]:
    rows = _pg_session_get(session_id, username)
    if rows is None:
        return None
    msgs = _pg_message_list(session_id, username)
    return SessionDetailResponse(
        session_id=session_id,
        messages=[
            SessionMessage(
                type=m["msg_type"],
                content=m["content"],
                rag_trace=m.get("rag_trace"),
            )
            for m in msgs
        ],
    )


def delete_session(session_id: str, username: str) -> bool:
    return _pg_session_delete(session_id, username)
