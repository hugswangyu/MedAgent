"""会话持久化：PostgreSQL 存储，支持按用户隔离。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from medrag.infrastructure.storage.postgres_client import (
    session_save as _pg_session_save,
    session_update as _pg_session_update,
    session_list as _pg_session_list,
    session_get as _pg_session_get,
    session_delete as _pg_session_delete,
    message_add as _pg_message_add,
    message_list as _pg_message_list,
)

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
) -> None:
    """向会话追加一条消息。"""
    if not username:
        raise ValueError("username is required for session writes")

    # Ensure session exists
    existing = _pg_session_get(session_id, username)
    if existing is None:
        if not _pg_session_save(session_id, username):
            raise PermissionError("session belongs to another user")
        msg_count = 1
    else:
        msg_count = existing.get("message_count", 0) + 1

    if not _pg_message_add(session_id, username, msg_type, content, rag_trace):
        raise PermissionError("session belongs to another user")
    if not _pg_session_update(session_id, username, msg_count):
        raise PermissionError("session belongs to another user")


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
