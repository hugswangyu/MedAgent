"""会话持久化：JSON 文件存储。

在 ``tmp_data/sessions.json`` 中维护所有会话及其消息。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .schemas import SessionSummary, SessionMessage, SessionDetailResponse

logger = logging.getLogger(__name__)

_STORAGE_FILE = os.path.join("tmp_data", "sessions.json")

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load() -> Dict[str, list]:
    try:
        with open(_STORAGE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: Dict[str, list]) -> None:
    folder = os.path.dirname(_STORAGE_FILE)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(_STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def add_message(
    session_id: str,
    msg_type: str,
    content: str,
    rag_trace: Optional[dict] = None,
) -> None:
    """向会话追加一条消息。"""
    sessions = _load()
    entry = {
        "type": msg_type,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if rag_trace:
        entry["rag_trace"] = rag_trace
    sessions.setdefault(session_id, []).append(entry)
    _save(sessions)


def get_sessions() -> List[SessionSummary]:
    sessions = _load()
    result = []
    for sid, msgs in sessions.items():
        updated = msgs[-1]["timestamp"] if msgs else ""
        result.append(
            SessionSummary(
                session_id=sid,
                message_count=len(msgs),
                updated_at=updated,
            )
        )
    result.sort(key=lambda s: s.updated_at, reverse=True)
    return result


def get_session(session_id: str) -> Optional[SessionDetailResponse]:
    sessions = _load()
    msgs = sessions.get(session_id)
    if msgs is None:
        return None
    return SessionDetailResponse(
        session_id=session_id,
        messages=[
            SessionMessage(
                type=m["type"],
                content=m["content"],
                rag_trace=m.get("rag_trace"),
            )
            for m in msgs
        ],
    )


def delete_session(session_id: str) -> bool:
    sessions = _load()
    if session_id not in sessions:
        return False
    del sessions[session_id]
    _save(sessions)
    return True
