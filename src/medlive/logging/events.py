"""JSONL 事件日志。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    """把运行事件追加到 JSONL 文件。"""

    def __init__(
        self,
        log_path: Path,
        *,
        room: str | None = None,
        room_id: str | None = None,
        job_id: str | None = None,
        agent_name: str = "my-agent",
    ) -> None:
        """绑定事件日志路径和会话元信息。"""

        self._log_path = log_path
        self._room = room
        self._room_id = room_id
        self._job_id = job_id
        self._agent_name = agent_name

    def append(self, event_name: str, payload: dict[str, Any]) -> None:
        """追加一条事件记录。"""

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            "room": self._room,
            "room_id": self._room_id,
            "job_id": self._job_id,
            "agent_name": self._agent_name,
            "payload": payload,
        }
        with self._log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
