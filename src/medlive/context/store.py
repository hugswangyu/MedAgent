"""提示词、历史、会话和知识库上下文文件存储。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from medlive.context.defaults import (
    DEFAULT_HISTORY_COMPRESS_PROMPT,
    DEFAULT_KNOWLEDGE_OVERVIEW_FALLBACK,
    DEFAULT_KNOWLEDGE_OVERVIEW_PROMPT,
    DEFAULT_SOUL,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
)
from medlive.runtime.paths import RuntimePaths, ensure_runtime_dirs

MessageRole = Literal["user", "assistant", "system"]


class ContextStore:
    """只负责上下文相关文件读写，不承担模型判断。"""

    def __init__(self, paths: RuntimePaths) -> None:
        """绑定运行路径。"""

        self.paths = paths
        self.voice_session_id: str | None = None
        self._ephemeral = False
        self._ephemeral_messages: list[dict[str, Any]] = []
        self._ephemeral_rag_context: list[dict[str, Any]] = []
        self._ephemeral_system_prompt = ""
        self._ephemeral_runtime_state: dict[str, Any] = {}

    def for_voice_session(self, session_id: str | None) -> ContextStore:
        """返回默认读写指定 voice session 文件的 store。"""

        scoped = ContextStore(self.paths)
        scoped.voice_session_id = self._safe_session_id(session_id) if session_id else None
        return scoped

    def for_ephemeral_session(self) -> ContextStore:
        """Return a process-only session store with no JSONL/history persistence."""

        scoped = ContextStore(self.paths)
        scoped._ephemeral = True
        return scoped

    def initialize(self, *, reset_session: bool = False) -> None:
        """初始化用户目录和默认提示词模板。"""

        ensure_runtime_dirs(self.paths)
        self._ensure_text_file(self.paths.system_prompt_template_file, DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self._ensure_text_file(self.paths.soul_file, DEFAULT_SOUL)
        self._ensure_text_file(self.paths.history_compress_prompt_file, DEFAULT_HISTORY_COMPRESS_PROMPT)
        self._ensure_text_file(self.paths.knowledge_overview_prompt_file, DEFAULT_KNOWLEDGE_OVERVIEW_PROMPT)
        self._ensure_text_file(self.paths.messages_file, "")
        self._ensure_text_file(self.paths.rag_context_file, "")
        self._ensure_text_file(self.paths.session_system_prompt_file, "")
        self._ensure_text_file(self.paths.runtime_state_file, "{}\n")
        if reset_session:
            self.clear_session()

    def clear_session(self) -> None:
        """清空当前通话数据。"""

        if self._ephemeral:
            self._ephemeral_messages.clear()
            self._ephemeral_rag_context.clear()
            self._ephemeral_system_prompt = ""
            self._ephemeral_runtime_state = {}
            return
        self._messages_file().write_text("", encoding="utf-8")
        self._rag_context_file().write_text("", encoding="utf-8")
        self.paths.session_system_prompt_file.write_text("", encoding="utf-8")
        self.write_runtime_state({})

    def clear_session_messages(self) -> None:
        """只清空当前通话消息。"""

        if self._ephemeral:
            self._ephemeral_messages.clear()
            return
        self._messages_file().write_text("", encoding="utf-8")

    def read_system_prompt_template(self) -> str:
        """读取系统提示词模板。"""

        return self._read_text(self.paths.system_prompt_template_file, DEFAULT_SYSTEM_PROMPT_TEMPLATE)

    def write_system_prompt_template(self, content: str) -> None:
        """写入系统提示词模板。"""

        self.paths.system_prompt_template_file.write_text(content.rstrip() + "\n", encoding="utf-8")

    def read_soul(self) -> str:
        """读取用户定义的 Agent 角色人格。"""

        return self._read_text(self.paths.soul_file, DEFAULT_SOUL)

    def write_soul(self, content: str) -> None:
        """写入用户定义的 Agent 角色人格。"""

        self.paths.soul_file.write_text(content.rstrip() + "\n", encoding="utf-8")

    def read_history_compress_prompt(self) -> str:
        """读取通话历史压缩提示词。"""

        return self._read_text(self.paths.history_compress_prompt_file, DEFAULT_HISTORY_COMPRESS_PROMPT)

    def write_history_compress_prompt(self, content: str) -> None:
        """写入通话历史压缩提示词。"""

        self.paths.history_compress_prompt_file.write_text(content.rstrip() + "\n", encoding="utf-8")

    def read_knowledge_overview_prompt(self) -> str:
        """读取知识库概览生成提示词。"""

        return self._read_text(self.paths.knowledge_overview_prompt_file, DEFAULT_KNOWLEDGE_OVERVIEW_PROMPT)

    def write_knowledge_overview_prompt(self, content: str) -> None:
        """写入知识库概览生成提示词。"""

        self.paths.knowledge_overview_prompt_file.write_text(content.rstrip() + "\n", encoding="utf-8")

    def read_session_system_prompt(self) -> str:
        """读取本次通话已经锁定的系统提示词。"""

        if self._ephemeral:
            return self._ephemeral_system_prompt
        return self._read_text(self.paths.session_system_prompt_file, "")

    def write_session_system_prompt(self, content: str) -> None:
        """写入本次通话已经锁定的系统提示词。"""

        if self._ephemeral:
            self._ephemeral_system_prompt = content.rstrip()
            return
        self.paths.session_system_prompt_file.write_text(content.rstrip() + "\n", encoding="utf-8")

    def append_message(
        self,
        *,
        role: MessageRole,
        content: str,
        turn_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """追加当前通话消息。"""

        text = content.strip()
        if not text:
            return
        record = {
            "timestamp": self._now_iso(),
            "session_id": self.voice_session_id,
            "role": role,
            "content": text,
            "turn_index": turn_index,
            "turn_id": f"turn_{turn_index}" if turn_index is not None else None,
            "metadata": metadata or {},
        }
        if self._ephemeral:
            self._ephemeral_messages.append(record)
        else:
            self._append_jsonl(self._messages_file(), record)

    def read_messages(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """读取当前通话消息。"""

        items = (
            list(self._ephemeral_messages)
            if self._ephemeral
            else self._read_jsonl(self._messages_file())
        )
        return items[-limit:] if limit and limit > 0 else items

    def append_rag_context(self, record: dict[str, Any]) -> None:
        """追加一轮 RAG 查询事实。"""

        turn_index = self._coerce_turn_index(record.get("turn_index"))
        stored = {
            "timestamp": self._now_iso(),
            "session_id": self.voice_session_id,
            "turn_id": (
                str(record.get("turn_id"))
                if record.get("turn_id")
                else (f"turn_{turn_index}" if turn_index is not None else None)
            ),
            **record,
        }
        if self._ephemeral:
            self._ephemeral_rag_context.append(stored)
        else:
            self._append_jsonl(self._rag_context_file(), stored)

    def read_rag_context(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """读取当前通话 RAG 查询事实。"""

        items = (
            list(self._ephemeral_rag_context)
            if self._ephemeral
            else self._read_jsonl(self._rag_context_file())
        )
        return items[-limit:] if limit and limit > 0 else items

    def read_voice_session_messages(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """读取指定 voice session 的消息。"""

        return self.for_voice_session(session_id).read_messages(limit=limit)

    def read_voice_session_rag_context(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """读取指定 voice session 的 RAG 查询事实。"""

        return self.for_voice_session(session_id).read_rag_context(limit=limit)

    def read_voice_session_turns(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """读取指定 voice session 的聚合 turns。"""

        return self.for_voice_session(session_id).read_session_turns(limit=limit)

    def read_session_turns(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """按轮次聚合当前通话消息和 RAG 证据。"""

        turns: dict[int, dict[str, Any]] = {}
        for message in self.read_messages():
            turn_index = self._coerce_turn_index(message.get("turn_index"))
            if turn_index is None:
                continue
            turn = self._ensure_turn(turns, turn_index)
            turn["messages"].append(message)
            role = message.get("role")
            if role == "user":
                turn["user_message"] = message
            elif role == "assistant":
                turn["assistant_message"] = message

        for record in self.read_rag_context():
            turn_index = self._coerce_turn_index(record.get("turn_index"))
            if turn_index is None:
                continue
            turn = self._ensure_turn(turns, turn_index)
            turn["rag_contexts"].append(record)

        items = [turns[index] for index in sorted(turns)]
        if limit and limit > 0:
            items = items[-limit:]
        for turn in items:
            contexts = turn["rag_contexts"]
            turn["rag"] = self._build_rag_summary(contexts[-1] if contexts else None)
        return items

    def history_file(self, kb_id: str) -> Path:
        """返回指定知识库历史文件路径。"""

        return self._history_dir(kb_id) / "history.jsonl"

    def read_history(self, kb_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """独立长期 history 已停用；保留签名仅用于迁移期显式失败安全。"""

        del kb_id, limit
        return []

    def append_history(self, kb_id: str, content: str) -> dict[str, Any]:
        """拒绝写入已停用的独立长期 history。"""

        del kb_id, content
        raise RuntimeError(
            "cannot persist history: LiveRAG independent history is disabled; "
            "PostgreSQL is the sole fact source"
        )

    def clear_history(self, kb_id: str) -> None:
        """拒绝修改已停用的独立长期 history。"""

        del kb_id
        raise RuntimeError("LiveRAG independent history is disabled; PostgreSQL is the sole fact source")

    def read_knowledge_overview(self, kb_id: str) -> str:
        """读取指定知识库的固定概览。"""

        self.ensure_knowledge_overview_default(kb_id)
        return self._read_text(self._overview_file(kb_id), DEFAULT_KNOWLEDGE_OVERVIEW_FALLBACK)

    def ensure_knowledge_overview_default(self, kb_id: str) -> None:
        """确保指定知识库至少有默认概览文件。"""

        if self._overview_file(kb_id).is_file():
            return
        self.write_knowledge_overview(
            kb_id,
            DEFAULT_KNOWLEDGE_OVERVIEW_FALLBACK,
            stale=True,
            reason="default_created",
            source="default",
        )

    def write_knowledge_overview(
        self,
        kb_id: str,
        content: str,
        *,
        stale: bool,
        reason: str | None = None,
        source: str = "context_model",
        source_job_id: str | None = None,
        raw_overview: dict[str, Any] | None = None,
    ) -> None:
        """写入指定知识库的固定概览和元数据。"""

        directory = self._context_kb_dir(kb_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._overview_file(kb_id).write_text(content.rstrip() + "\n", encoding="utf-8")
        self._overview_meta_file(kb_id).write_text(
            json.dumps(
                {
                    "kb_id": kb_id,
                    "updated_at": self._now_iso(),
                    "stale": stale,
                    "reason": reason,
                    "source": source,
                    "source_job_id": source_job_id,
                    "raw_summary": self._raw_overview_summary(raw_overview),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def read_knowledge_overview_meta(self, kb_id: str) -> dict[str, Any]:
        """读取指定知识库概览元数据。"""

        self.ensure_knowledge_overview_default(kb_id)
        try:
            data = json.loads(self._overview_meta_file(kb_id).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"kb_id": kb_id, "stale": True, "reason": "missing"}
        return data if isinstance(data, dict) else {"kb_id": kb_id, "stale": True, "reason": "invalid"}

    def mark_knowledge_overview_stale(self, kb_id: str, *, reason: str) -> None:
        """把指定知识库概览标记为过期。"""

        current = self.read_knowledge_overview(kb_id)
        self.write_knowledge_overview(
            kb_id,
            current,
            stale=True,
            reason=reason,
            source="stale_marker",
            source_job_id=None,
        )

    def read_runtime_state(self) -> dict[str, Any]:
        """读取当前运行状态。"""

        if self._ephemeral:
            return dict(self._ephemeral_runtime_state)
        try:
            data = json.loads(self.paths.runtime_state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def write_runtime_state(self, state: dict[str, Any]) -> None:
        """写入当前运行状态。"""

        if self._ephemeral:
            self._ephemeral_runtime_state = dict(state)
            return
        self.paths.runtime_state_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _history_dir(self, kb_id: str) -> Path:
        directory = self.paths.history_dir / self._safe_kb_id(kb_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _context_kb_dir(self, kb_id: str) -> Path:
        directory = self.paths.context_dir / self._safe_kb_id(kb_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _overview_file(self, kb_id: str) -> Path:
        return self._context_kb_dir(kb_id) / "knowledge_overview.md"

    def _overview_meta_file(self, kb_id: str) -> Path:
        return self._context_kb_dir(kb_id) / "knowledge_overview_meta.json"

    def voice_session_dir(self, session_id: str) -> Path:
        """返回指定 voice session 的上下文文件目录。"""

        directory = self.paths.voice_sessions_dir / self._safe_session_id(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _messages_file(self) -> Path:
        if not self.voice_session_id:
            return self.paths.messages_file
        return self.voice_session_dir(self.voice_session_id) / "messages.jsonl"

    def _rag_context_file(self) -> Path:
        if not self.voice_session_id:
            return self.paths.rag_context_file
        return self.voice_session_dir(self.voice_session_id) / "rag_context.jsonl"

    @staticmethod
    def _raw_overview_summary(raw_overview: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(raw_overview, dict):
            return {}
        summary = raw_overview.get("summary")
        return summary if isinstance(summary, dict) else {}

    @staticmethod
    def _safe_kb_id(kb_id: str) -> str:
        clean = kb_id.strip() or "default"
        if not all(char.isalnum() or char in {"_", "-"} for char in clean):
            raise ValueError(f"invalid kb_id: {kb_id}")
        return clean

    @staticmethod
    def _safe_session_id(session_id: str | None) -> str:
        clean = (session_id or "").strip()
        if not clean or not all(char.isalnum() or char in {"_", "-"} for char in clean):
            raise ValueError(f"invalid session_id: {session_id}")
        return clean

    @staticmethod
    def _ensure_turn(turns: dict[int, dict[str, Any]], turn_index: int) -> dict[str, Any]:
        if turn_index not in turns:
            turns[turn_index] = {
                "turn_index": turn_index,
                "turn_id": f"turn_{turn_index}",
                "messages": [],
                "user_message": None,
                "assistant_message": None,
                "rag": ContextStore._build_rag_summary(None),
                "rag_contexts": [],
            }
        return turns[turn_index]

    @staticmethod
    def _build_rag_summary(record: dict[str, Any] | None) -> dict[str, Any]:
        if record is None:
            return {
                "status": "not_queried",
                "queried": False,
                "hit": None,
                "has_context": False,
                "query": None,
                "effective_query": None,
                "request_id": None,
                "latency_ms": None,
                "cache_hit": False,
                "evidence_documents": [],
                "evidence_chunks": [],
                "evidence_count": 0,
                "no_evidence_reason": None,
                "error": None,
                "context_preview": "",
                "kb_id": None,
                "kb_name": None,
            }

        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        error = record.get("error")
        hit = record.get("hit")
        has_context = bool(record.get("has_context"))
        if error:
            status = "failed"
        elif has_context or hit is True:
            status = "hit"
        else:
            status = "miss"
        return {
            "status": status,
            "queried": True,
            "hit": hit,
            "has_context": has_context,
            "query": record.get("query"),
            "effective_query": record.get("effective_query"),
            "request_id": record.get("request_id"),
            "latency_ms": metrics.get("latency_ms"),
            "cache_hit": bool(metrics.get("cache_hit")),
            "evidence_documents": record.get("evidence_documents") or [],
            "evidence_chunks": record.get("evidence_chunks") or [],
            "evidence_count": record.get("evidence_count") or 0,
            "no_evidence_reason": record.get("no_evidence_reason"),
            "error": error,
            "context_preview": record.get("context_preview") or "",
            "kb_id": record.get("kb_id"),
            "kb_name": record.get("kb_name"),
        }

    @staticmethod
    def _coerce_turn_index(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    @staticmethod
    def _ensure_text_file(path: Path, default: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(default.rstrip() + "\n", encoding="utf-8")

    @staticmethod
    def _read_text(path: Path, default: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return default

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        items: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                items.append(value)
        return items

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
