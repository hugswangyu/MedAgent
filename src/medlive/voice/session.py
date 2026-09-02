"""Voice session storage, token issuance, and lifecycle orchestration."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from livekit import api as livekit_api

from medlive.api.rag_gateway import envelope
from medlive.config.settings import AppSettings
from medlive.context.store import ContextStore
from medlive.control_plane import ControlPlaneClient, ControlPlaneError
from medlive.rag.metadata_store import MetadataStore

VoiceSessionStatus = Literal["created", "active", "ending", "ended", "expired", "failed"]
_ACTIVE_STATUSES = {"created", "active", "ending"}
_TERMINAL_STATUSES = {"ended", "expired", "failed"}


@dataclass(frozen=True)
class TokenResult:
    """LiveKit token issuance result."""

    token: str
    expires_at: str


class VoiceSessionStore:
    """SQLite-backed voice session state store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser()

    def initialize(self) -> None:
        """Create voice session tables."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS voice_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    client_id TEXT,
                    client_type TEXT NOT NULL,
                    kb_id TEXT NOT NULL,
                    kb_name TEXT NOT NULL,
                    room_name TEXT NOT NULL,
                    participant_identity TEXT NOT NULL,
                    agent_job_id TEXT,
                    status TEXT NOT NULL,
                    token_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    last_seen_at TEXT,
                    history_compacted_at TEXT,
                    session_prompt_chars INTEGER,
                    history_count INTEGER,
                    metadata_json TEXT,
                    error_msg TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_voice_sessions_status_created
                    ON voice_sessions(status, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_voice_sessions_room
                    ON voice_sessions(room_name);
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(voice_sessions)").fetchall()
            }
            if "user_id" not in columns:
                conn.execute(
                    "ALTER TABLE voice_sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_voice_sessions_user_created "
                "ON voice_sessions(user_id, created_at DESC)"
            )

    def create_session(
        self,
        *,
        session_id: str,
        user_id: str,
        client_id: str | None,
        client_type: str,
        kb_id: str,
        kb_name: str,
        room_name: str,
        participant_identity: str,
        token_expires_at: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a newly created voice session."""

        now = self._now_iso()
        row = {
            "session_id": session_id,
            "user_id": user_id,
            "client_id": client_id or "",
            "client_type": client_type,
            "kb_id": kb_id,
            "kb_name": kb_name,
            "room_name": room_name,
            "participant_identity": participant_identity,
            "agent_job_id": None,
            "status": "created",
            "token_expires_at": token_expires_at,
            "created_at": now,
            "started_at": None,
            "ended_at": None,
            "last_seen_at": now,
            "history_compacted_at": None,
            "session_prompt_chars": None,
            "history_count": None,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "error_msg": None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO voice_sessions(
                    session_id, user_id, client_id, client_type, kb_id, kb_name, room_name,
                    participant_identity, agent_job_id, status, token_expires_at,
                    created_at, started_at, ended_at, last_seen_at, history_compacted_at,
                    session_prompt_chars, history_count, metadata_json, error_msg
                ) VALUES (
                    :session_id, :user_id, :client_id, :client_type, :kb_id, :kb_name, :room_name,
                    :participant_identity, :agent_job_id, :status, :token_expires_at,
                    :created_at, :started_at, :ended_at, :last_seen_at, :history_compacted_at,
                    :session_prompt_chars, :history_count, :metadata_json, :error_msg
                )
                """,
                row,
            )
        return self.get_session(session_id)

    def get_session(
        self, session_id: str, user_id: str | None = None
    ) -> dict[str, Any]:
        """Read one session by id."""

        with self._connect() as conn:
            query = "SELECT * FROM voice_sessions WHERE session_id = ?"
            params: list[Any] = [session_id]
            if user_id is not None:
                query += " AND user_id = ?"
                params.append(user_id)
            row = conn.execute(query, params).fetchone()
        if row is None:
            raise KeyError(f"voice session not found: {session_id}")
        return self._public_from_row(row)

    def active_session(self, user_id: str | None = None) -> dict[str, Any] | None:
        """Return the latest non-terminal voice session if any."""

        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        with self._connect() as conn:
            query = f"""
                SELECT * FROM voice_sessions
                WHERE status IN ({placeholders})
            """
            params: list[Any] = list(_ACTIVE_STATUSES)
            if user_id is not None:
                query += " AND user_id = ?"
                params.append(user_id)
            query += """
                ORDER BY created_at DESC
                LIMIT 1
            """
            row = conn.execute(query, params).fetchone()
        return self._public_from_row(row) if row is not None else None

    def mark_active(
        self,
        session_id: str,
        *,
        agent_job_id: str | None,
        session_prompt_chars: int,
        history_count: int,
        room_name: str | None = None,
    ) -> dict[str, Any]:
        """Mark a session as bound to an Agent job."""

        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE voice_sessions
                SET status = 'active',
                    agent_job_id = ?,
                    started_at = COALESCE(started_at, ?),
                    last_seen_at = ?,
                    session_prompt_chars = ?,
                    history_count = ?,
                    room_name = COALESCE(?, room_name)
                WHERE session_id = ? AND status IN ('created', 'active')
                """,
                (agent_job_id, now, now, session_prompt_chars, history_count, room_name, session_id),
            )
        return self.get_session(session_id)

    def touch(self, session_id: str) -> None:
        """Update session last seen timestamp."""

        with self._connect() as conn:
            conn.execute(
                "UPDATE voice_sessions SET last_seen_at = ? WHERE session_id = ?",
                (self._now_iso(), session_id),
            )

    def mark_ending(self, session_id: str) -> dict[str, Any]:
        """Move active session to ending. Terminal sessions are returned unchanged."""

        current = self.get_session(session_id)
        if current["status"] in _TERMINAL_STATUSES:
            return current
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE voice_sessions
                SET status = 'ending', last_seen_at = ?
                WHERE session_id = ? AND status NOT IN ('ended', 'expired', 'failed')
                """,
                (self._now_iso(), session_id),
            )
        return self.get_session(session_id)

    def mark_ended(
        self,
        session_id: str,
        *,
        history_compacted_at: str | None = None,
        error_msg: str | None = None,
    ) -> dict[str, Any]:
        """Mark session ended without overwriting existing compact timestamp."""

        now = self._now_iso()
        current = self.get_session(session_id)
        compacted_at = current.get("history_compacted_at") or history_compacted_at
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE voice_sessions
                SET status = 'ended',
                    ended_at = COALESCE(ended_at, ?),
                    last_seen_at = ?,
                    history_compacted_at = COALESCE(history_compacted_at, ?),
                    error_msg = COALESCE(?, error_msg)
                WHERE session_id = ?
                """,
                (now, now, compacted_at, error_msg, session_id),
            )
        return self.get_session(session_id)

    def mark_expired(self, session_id: str, *, reason: str) -> dict[str, Any]:
        """Mark session expired."""

        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE voice_sessions
                SET status = 'expired', ended_at = COALESCE(ended_at, ?),
                    last_seen_at = ?, error_msg = COALESCE(error_msg, ?)
                WHERE session_id = ? AND status NOT IN ('ended', 'expired', 'failed')
                """,
                (now, now, reason, session_id),
            )
        return self.get_session(session_id)

    def mark_failed(self, session_id: str, *, error_msg: str) -> dict[str, Any]:
        """Mark session failed."""

        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE voice_sessions
                SET status = 'failed', ended_at = COALESCE(ended_at, ?),
                    last_seen_at = ?, error_msg = ?
                WHERE session_id = ?
                """,
                (now, now, error_msg, session_id),
            )
        return self.get_session(session_id)

    def sessions_for_cleanup(self) -> list[dict[str, Any]]:
        """Return non-terminal sessions for cleanup."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM voice_sessions
                WHERE status IN ('created', 'active', 'ending')
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [self._public_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _public_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        metadata_json = str(row["metadata_json"] or "{}")
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"] or "",
            "client_id": row["client_id"] or "",
            "client_type": row["client_type"],
            "kb_id": row["kb_id"],
            "kb_name": row["kb_name"],
            "room_name": row["room_name"],
            "participant_identity": row["participant_identity"],
            "agent_job_id": row["agent_job_id"],
            "status": row["status"],
            "token_expires_at": row["token_expires_at"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "last_seen_at": row["last_seen_at"],
            "history_compacted_at": row["history_compacted_at"],
            "session_prompt_chars": row["session_prompt_chars"],
            "history_count": row["history_count"],
            "metadata": metadata if isinstance(metadata, dict) else {},
            "error_msg": row["error_msg"],
        }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class VoiceSessionService:
    """High-level voice session API service."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        store: ContextStore,
        metadata_store: MetadataStore,
        control_plane: ControlPlaneClient,
    ) -> None:
        self.settings = settings
        self.store = store
        self.metadata_store = metadata_store
        self.control_plane = control_plane

    async def create_session(
        self,
        *,
        user_id: str,
        kb_id: str,
        client_id: str | None,
        client_type: str,
        access_token: str,
    ) -> dict[str, Any]:
        """Create a business voice session and issue a LiveKit join token."""

        if client_type not in {"android", "web", "test"}:
            raise VoiceSessionError(400, "InvalidRequest", "client_type must be android, web or test")
        try:
            await self.control_plane.get_knowledge_base(access_token, kb_id)
            kb = self.metadata_store.public_knowledge_base_detail(kb_id)
        except (KeyError, ControlPlaneError) as exc:
            raise VoiceSessionError(404, "KnowledgeBaseNotFound", f"knowledge base not found: {kb_id}") from exc

        session_id = f"vs_{uuid.uuid4().hex[:16]}"
        room_name = f"medlive_{session_id}"
        identity = f"{client_type}_{uuid.uuid4().hex[:10]}"
        room_metadata = {"session_id": session_id, "binding_version": 1}
        token = self.issue_token(room_name=room_name, identity=identity)
        try:
            session = await self.control_plane.create_voice_session(
                access_token,
                {
                    "session_id": session_id,
                    "knowledge_base_id": str(kb["kb_id"]),
                    "room_name": room_name,
                    "client_id": client_id,
                    "client_type": client_type,
                    "participant_identity": identity,
                    "token_expires_at": token.expires_at,
                    "metadata": room_metadata,
                },
            )
        except ControlPlaneError as exc:
            error_type = "ActiveSessionExists" if exc.status_code == 409 else "VoiceSessionCreateFailed"
            raise VoiceSessionError(exc.status_code, error_type, str(exc)) from exc
        room_metadata["binding_version"] = int(session["binding_version"])
        dispatch_metadata = {
            **room_metadata,
            "worker_bootstrap_token": session.pop("worker_bootstrap_token"),
        }
        try:
            await self._create_livekit_room(
                room_name=room_name,
                room_metadata=room_metadata,
                dispatch_metadata=dispatch_metadata,
            )
        except Exception:
            with suppress(ControlPlaneError):
                await self.control_plane.end_voice_session(access_token, session_id)
            raise
        session["kb_name"] = str(kb["name"])
        session["token_expires_at"] = token.expires_at
        session["metadata"] = room_metadata
        return {
            **self.public_session_detail(session),
            "token": token.token,
            "identity": identity,
            "livekit_url": self.public_livekit_url,
        }

    async def get_session(
        self, session_id: str, *, user_id: str, access_token: str
    ) -> dict[str, Any]:
        """Return public session detail."""

        try:
            session = await self.control_plane.get_voice_session(access_token, session_id)
        except ControlPlaneError as exc:
            raise KeyError(session_id) from exc
        self._attach_kb_name(session)
        return self.public_session_detail(session)

    async def refresh_token(
        self, session_id: str, *, user_id: str, access_token: str
    ) -> dict[str, Any]:
        """Refresh the LiveKit token for a created/active session."""

        try:
            session = await self.control_plane.get_voice_session(access_token, session_id)
        except ControlPlaneError as exc:
            raise KeyError(session_id) from exc
        status = str(session["status"])
        if status in {"ended", "expired", "failed"}:
            raise VoiceSessionError(410, "VoiceSessionEnded", "voice session is already closed")
        if status not in {"created", "active"}:
            raise VoiceSessionError(422, "InvalidStateTransition", f"cannot refresh token for {status} session")
        token = self.issue_token(
            room_name=session["room_name"],
            identity=session["participant_identity"],
        )
        return {
            "session_id": session_id,
            "room_name": session["room_name"],
            "identity": session["participant_identity"],
            "livekit_url": self.public_livekit_url,
            "token": token.token,
            "token_expires_at": token.expires_at,
        }

    async def end_session(
        self,
        session_id: str,
        *,
        reason: str = "api",
        user_id: str | None = None,
        access_token: str,
    ) -> dict[str, Any]:
        """Idempotently end a voice session without local long-term history."""

        try:
            session = await self.control_plane.get_voice_session(access_token, session_id)
        except ControlPlaneError as exc:
            raise KeyError(session_id) from exc
        self._attach_kb_name(session)
        if session["status"] == "ended":
            return self.public_session_detail(session)
        if session["status"] in {"expired", "failed"}:
            return self.public_session_detail(session)

        await self.control_plane.end_voice_session(access_token, session_id)
        ended = {
            **session,
            "status": "ended",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        detail = self.public_session_detail(ended)
        detail["history_compaction"] = {
            "updated": False,
            "reason": "medlive_independent_history_disabled",
        }
        detail["end_reason"] = reason
        return detail

    def issue_token(self, *, room_name: str, identity: str) -> TokenResult:
        """Issue a room-scoped LiveKit client token."""

        if not self.settings.api.livekit_api_key or not self.settings.api.livekit_api_secret:
            raise VoiceSessionError(500, "LiveKitTokenFailed", "LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not configured")
        ttl_s = max(int(self.settings.api.voice_session_token_ttl_s), 60)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_s)
        token = (
            livekit_api.AccessToken(self.settings.api.livekit_api_key, self.settings.api.livekit_api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(
                livekit_api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .with_ttl(timedelta(seconds=ttl_s))
            .to_jwt()
        )
        return TokenResult(token=token, expires_at=expires_at.isoformat())

    def public_session_detail(self, session: dict[str, Any]) -> dict[str, Any]:
        """Shape a DB row for public API responses."""

        return {
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "client_id": session.get("client_id") or "",
            "client_type": session["client_type"],
            "status": session["status"],
            "kb": {
                "kb_id": session.get("knowledge_base_id") or session.get("kb_id"),
                "name": session.get("kb_name") or "",
            },
            "livekit": {
                "livekit_url": self.public_livekit_url,
                "room_name": session["room_name"],
                "participant_identity": session["participant_identity"],
                "agent_job_id": session.get("livekit_job_id") or session.get("agent_job_id"),
                "token_expires_at": session.get("token_expires_at"),
            },
            "agent": {
                "bound": bool(session.get("livekit_job_id") or session.get("agent_job_id")),
                "started_at": session.get("started_at"),
                "session_prompt_chars": session.get("session_prompt_chars"),
                "history_count": session.get("history_count"),
            },
            "lifecycle": {
                "created_at": session.get("created_at"),
                "started_at": session.get("started_at"),
                "ended_at": session.get("ended_at"),
                "last_seen_at": session.get("last_seen_at"),
                "history_compacted_at": session.get("history_compacted_at"),
            },
            "metadata": session.get("metadata") or {},
            "error": {"message": session["error_msg"]} if session.get("error_msg") else None,
        }

    async def paginate_turns(
        self,
        session_id: str,
        *,
        user_id: str,
        limit: int,
        cursor: int | None,
        access_token: str,
    ) -> dict[str, Any]:
        """Return one page of turns for a voice session."""

        try:
            await self.control_plane.get_voice_session(access_token, session_id)
        except ControlPlaneError as exc:
            raise KeyError(session_id) from exc
        items = self.store.read_voice_session_turns(session_id)
        return self._paginate(items, limit=limit, cursor=cursor)

    async def paginate_rag_context(
        self,
        session_id: str,
        *,
        user_id: str,
        limit: int,
        cursor: int | None,
        access_token: str,
    ) -> dict[str, Any]:
        """Return one page of RAG records for a voice session."""

        try:
            await self.control_plane.get_voice_session(access_token, session_id)
        except ControlPlaneError as exc:
            raise KeyError(session_id) from exc
        items = self.store.read_voice_session_rag_context(session_id)
        return self._paginate(items, limit=limit, cursor=cursor)

    async def cleanup_stale_sessions(self) -> None:
        """Expire stale sessions based on configured TTLs."""
        try:
            await self.control_plane.cleanup_stale_voice_sessions()
        except ControlPlaneError as exc:
            raise VoiceSessionError(
                exc.status_code, "VoiceSessionCleanupFailed", str(exc)
            ) from exc

    @property
    def public_livekit_url(self) -> str:
        return self.settings.api.livekit_public_url or self.settings.voice.livekit_url

    async def _create_livekit_room(
        self,
        *,
        room_name: str,
        room_metadata: dict[str, Any],
        dispatch_metadata: dict[str, Any],
    ) -> None:
        """Best-effort room creation with metadata for Agent binding."""

        if not self.settings.voice.livekit_url:
            raise VoiceSessionError(
                500, "LiveKitConfigMissing", "LIVEKIT_URL is not configured"
            )
        if not self.settings.api.livekit_api_key or not self.settings.api.livekit_api_secret:
            raise VoiceSessionError(
                500,
                "LiveKitConfigMissing",
                "LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not configured",
            )
        metadata_raw = json.dumps(room_metadata, ensure_ascii=False)
        dispatch_raw = json.dumps(dispatch_metadata, ensure_ascii=False)
        try:
            lkapi = livekit_api.LiveKitAPI(
                url=self.settings.voice.livekit_url,
                api_key=self.settings.api.livekit_api_key,
                api_secret=self.settings.api.livekit_api_secret,
            )
            try:
                await lkapi.room.create_room(
                    livekit_api.CreateRoomRequest(
                        name=room_name,
                        metadata=metadata_raw,
                        agents=[
                            livekit_api.RoomAgentDispatch(
                                agent_name="my-agent",
                                metadata=dispatch_raw,
                            )
                        ],
                    )
                )
            finally:
                await lkapi.aclose()
        except Exception as exc:
            raise VoiceSessionError(500, "VoiceSessionCreateFailed", str(exc)) from exc

    def _attach_kb_name(self, session: dict[str, Any]) -> None:
        kb_id = str(session.get("knowledge_base_id") or session.get("kb_id") or "")
        try:
            session["kb_name"] = str(
                self.metadata_store.public_knowledge_base_detail(kb_id)["name"]
            )
        except KeyError:
            session["kb_name"] = ""

    @staticmethod
    def _paginate(items: list[dict[str, Any]], *, limit: int, cursor: int | None) -> dict[str, Any]:
        safe_limit = min(max(limit, 1), 200)
        total = len(items)
        start = max(total - safe_limit, 0) if cursor is None else min(max(cursor, 0), total)
        end = min(start + safe_limit, total)
        next_cursor = end if end < total else None
        return {
            "items": items[start:end],
            "next_cursor": next_cursor,
            "has_more": next_cursor is not None,
            "total": total,
            "order": "asc",
        }

    @staticmethod
    def _age_s(value: Any, now: datetime) -> float:
        if not value:
            return 0.0
        try:
            timestamp = datetime.fromisoformat(str(value))
        except ValueError:
            return 0.0
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max((now - timestamp).total_seconds(), 0.0)


class VoiceSessionError(Exception):
    """Error that maps cleanly to a management API envelope."""

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message

    def to_response(self) -> tuple[int, dict[str, Any]]:
        return self.status_code, envelope(
            status="error",
            data=None,
            error={"type": self.error_type, "message": self.message},
        )
