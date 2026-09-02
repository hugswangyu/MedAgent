"""阶段 1 身份、语音会话与审计 PostgreSQL 仓储。"""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import psycopg2.extras

from medrag.infrastructure.storage.postgres_client import get_conn
from medrag.memory.controlled import (
    build_session_summary,
    candidate_key,
    extract_medical_fact_candidates,
)


@dataclass(frozen=True)
class UserRecord:
    """数据库用户记录；密码字段只在认证模块内部使用。"""

    user_id: str
    username: str
    normalized_username: str
    password_hash: str
    is_admin: bool
    status: str
    token_version: int


class UserMigrationConflictError(RuntimeError):
    """PostgreSQL 中存在规范化后冲突的用户名。"""

    def __init__(self, conflicts: list[dict[str, Any]]) -> None:
        super().__init__("normalized username conflicts require manual resolution")
        self.conflicts = conflicts


def normalize_username(username: str) -> str:
    return unicodedata.normalize("NFKC", username).strip().casefold()


def ensure_schema() -> None:
    """幂等安装阶段 1/2 schema。"""

    scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
    with get_conn() as conn:
        with conn.cursor() as cur:
            for name in (
                "phase1_identity_data.sql",
                "phase2_safe_voice.sql",
                "phase3_memory.sql",
            ):
                cur.execute((scripts_dir / name).read_text(encoding="utf-8"))
    _normalize_existing_users()


def _normalize_existing_users() -> None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT user_id, username FROM users ORDER BY created_at, user_id")
            rows = [dict(row) for row in cur.fetchall()]
            grouped: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                normalized = normalize_username(str(row["username"]))
                grouped.setdefault(normalized, []).append(
                    {"user_id": str(row["user_id"]), "username": str(row["username"])}
                )
            conflicts = [
                {"normalized_username": key, "users": values}
                for key, values in grouped.items()
                if len(values) > 1
            ]
            if conflicts:
                raise UserMigrationConflictError(conflicts)
            for normalized, values in grouped.items():
                cur.execute(
                    "UPDATE users SET normalized_username = %s WHERE user_id = %s",
                    (normalized, values[0]["user_id"]),
                )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_normalized_username "
                "ON users(normalized_username)"
            )
            cur.execute("ALTER TABLE users ALTER COLUMN normalized_username SET NOT NULL")


def import_legacy_users(users: Iterable[tuple[str, str, bool]]) -> int:
    """只导入缺失的旧用户；绝不改写已有 PostgreSQL 身份。"""

    imported = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for username, password_hash, is_admin in users:
                cur.execute(
                    """
                    INSERT INTO users(
                        user_id, username, normalized_username, password_hash, is_admin
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (normalized_username) DO NOTHING
                    """,
                    (
                        str(uuid.uuid4()),
                        username,
                        normalize_username(username),
                        password_hash,
                        is_admin,
                    ),
                )
                imported += max(cur.rowcount, 0)
    return imported


def get_user_by_username(username: str) -> UserRecord | None:
    return _get_user("normalized_username = %s", normalize_username(username))


def get_user_by_id(user_id: str) -> UserRecord | None:
    return _get_user("user_id = %s", user_id)


def _get_user(predicate: str, value: str) -> UserRecord | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, username, normalized_username, password_hash, "
                f"is_admin, status, token_version FROM users WHERE {predicate}",
                (value,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return UserRecord(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        normalized_username=str(row["normalized_username"]),
        password_hash=str(row["password_hash"]),
        is_admin=bool(row["is_admin"]),
        status=str(row["status"]),
        token_version=int(row["token_version"]),
    )


def create_user(username: str, password_hash: str, is_admin: bool) -> UserRecord | None:
    user_id = str(uuid.uuid4())
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO users(
                    user_id, username, normalized_username, password_hash, is_admin
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (normalized_username) DO NOTHING
                RETURNING user_id, username, normalized_username, password_hash,
                          is_admin, status, token_version
                """,
                (
                    user_id,
                    username,
                    normalize_username(username),
                    password_hash,
                    is_admin,
                ),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return UserRecord(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        normalized_username=str(row["normalized_username"]),
        password_hash=str(row["password_hash"]),
        is_admin=bool(row["is_admin"]),
        status=str(row["status"]),
        token_version=int(row["token_version"]),
    )


def register_knowledge_base(*, kb_id: str, owner_user_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO knowledge_base_ownership(kb_id, owner_user_id)
                VALUES (%s, %s)
                ON CONFLICT (kb_id) DO UPDATE
                SET status = 'active', updated_at = NOW()
                WHERE knowledge_base_ownership.owner_user_id = EXCLUDED.owner_user_id
                RETURNING *
                """,
                (kb_id, owner_user_id),
            )
            row = cur.fetchone()
    if row is None:
        raise PermissionError("knowledge base is owned by another user")
    return {**dict(row), "owner_user_id": str(row["owner_user_id"])}


def get_owned_knowledge_base(kb_id: str, owner_user_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM knowledge_base_ownership
                WHERE kb_id = %s AND owner_user_id = %s AND status = 'active'
                """,
                (kb_id, owner_user_id),
            )
            row = cur.fetchone()
    return {**dict(row), "owner_user_id": str(row["owner_user_id"])} if row else None


def list_owned_knowledge_bases(owner_user_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM knowledge_base_ownership
                WHERE owner_user_id = %s AND status = 'active'
                ORDER BY created_at
                """,
                (owner_user_id,),
            )
            rows = cur.fetchall()
    return [{**dict(row), "owner_user_id": str(row["owner_user_id"])} for row in rows]


def set_knowledge_base_status(
    *, kb_id: str, owner_user_id: str, status: str
) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE knowledge_base_ownership
                SET status = %s, updated_at = NOW()
                WHERE kb_id = %s AND owner_user_id = %s
                """,
                (status, kb_id, owner_user_id),
            )
            return cur.rowcount == 1


def consume_worker_nonce(
    *,
    token_jti: str,
    nonce: str,
    user_id: str,
    session_id: str,
    request_hash: str,
    expires_at: datetime,
) -> bool:
    """原子消费一次 nonce；重复 token_jti + nonce 返回 False。"""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM worker_request_nonces WHERE expires_at <= NOW()")
            cur.execute(
                """
                INSERT INTO worker_request_nonces(
                    token_jti, nonce, user_id, session_id, request_hash, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (token_jti, nonce) DO NOTHING
                """,
                (token_jti, nonce, user_id, session_id, request_hash, expires_at),
            )
            return cur.rowcount == 1


def create_voice_session_binding(
    *,
    user_id: str,
    session_id: str,
    knowledge_base_id: str,
    room_name: str,
    client_id: str | None = None,
    client_type: str = "web",
    participant_identity: str | None = None,
    token_expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    lease_seconds: int = 120,
) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            _expire_stale_voice_sessions(cur)
            cur.execute(
                """
                INSERT INTO voice_sessions(
                    session_id, user_id, knowledge_base_id, room_name, client_id,
                    client_type, participant_identity, token_expires_at, metadata,
                    lease_expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                        NOW() + (%s * INTERVAL '1 second'))
                RETURNING *
                """,
                (
                    session_id,
                    user_id,
                    knowledge_base_id,
                    room_name,
                    client_id,
                    client_type,
                    participant_identity,
                    token_expires_at,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    lease_seconds,
                ),
            )
            row = cur.fetchone()
            cur.execute(
                """
                INSERT INTO conversation_sessions(
                    session_type, session_id, user_id
                ) VALUES ('voice', %s, %s)
                ON CONFLICT (session_type, session_id) DO UPDATE SET
                    user_id = conversation_sessions.user_id
                WHERE conversation_sessions.user_id = EXCLUDED.user_id
                """,
                (session_id, user_id),
            )
            if cur.rowcount != 1:
                raise PermissionError("unified voice session belongs to another user")
    return _voice_binding(row)


def get_voice_session_binding(
    session_id: str, user_id: str | None = None
) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = "SELECT * FROM voice_sessions WHERE session_id = %s"
            params: list[Any] = [session_id]
            if user_id is not None:
                query += " AND user_id = %s"
                params.append(user_id)
            cur.execute(query, params)
            row = cur.fetchone()
    return _voice_binding(row) if row else None


def claim_voice_session_binding(
    *,
    session_id: str,
    expected_version: int,
    room_name: str,
    livekit_job_id: str,
    lease_seconds: int = 120,
) -> dict[str, Any] | None:
    """以 binding_version + 空 job 条件执行一次 LiveKit job CAS。"""

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE voice_sessions
                SET livekit_job_id = %s, status = 'active',
                    binding_version = binding_version + 1, updated_at = NOW(),
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second')
                WHERE session_id = %s
                  AND binding_version = %s
                  AND room_name = %s
                  AND status = 'created'
                  AND livekit_job_id IS NULL
                RETURNING *
                """,
                (
                    livekit_job_id,
                    lease_seconds,
                    session_id,
                    expected_version,
                    room_name,
                ),
            )
            row = cur.fetchone()
    return _voice_binding(row) if row else None


def validate_claimed_worker_binding(
    *,
    session_id: str,
    user_id: str,
    knowledge_base_id: str,
    livekit_job_id: str,
) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM voice_sessions
                WHERE session_id = %s AND user_id = %s AND knowledge_base_id = %s
                  AND status = 'active' AND livekit_job_id = %s
                  AND lease_expires_at > NOW()
                """,
                (session_id, user_id, knowledge_base_id, livekit_job_id),
            )
            return cur.fetchone() is not None


def renew_voice_session_lease(
    *,
    session_id: str,
    user_id: str,
    knowledge_base_id: str,
    livekit_job_id: str,
    lease_seconds: int = 120,
) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE voice_sessions
                SET lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE session_id = %s AND user_id = %s AND knowledge_base_id = %s
                  AND livekit_job_id = %s AND status = 'active'
                  AND lease_expires_at > NOW()
                """,
                (
                    lease_seconds,
                    session_id,
                    user_id,
                    knowledge_base_id,
                    livekit_job_id,
                ),
            )
            return cur.rowcount == 1


def expire_stale_voice_sessions() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            return _expire_stale_voice_sessions(cur)


def _expire_stale_voice_sessions(cur: Any) -> int:
    cur.execute(
        """
        UPDATE voice_sessions
        SET status = 'expired', ended_at = COALESCE(ended_at, NOW()),
            updated_at = NOW(), binding_version = binding_version + 1
        WHERE status IN ('created', 'active')
          AND lease_expires_at <= NOW()
        """
    )
    return max(cur.rowcount, 0)


def end_voice_session_binding(*, session_id: str, user_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE voice_sessions
                SET status = 'ended', ended_at = NOW(), updated_at = NOW(),
                    binding_version = binding_version + 1
                WHERE session_id = %s AND user_id = %s
                """,
                (session_id, user_id),
            )
            ended = cur.rowcount == 1
            if ended:
                cur.execute(
                    """
                    UPDATE conversation_sessions
                    SET ended_at = COALESCE(ended_at, NOW())
                    WHERE session_type = 'voice' AND session_id = %s
                      AND user_id = %s
                    """,
                    (session_id, user_id),
                )
            return ended


def _voice_binding(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["user_id"] = str(data["user_id"])
    data["kb_id"] = data["knowledge_base_id"]
    return data


def record_capability_event(
    *,
    user_id: str,
    session_id: str,
    turn_id: str,
    action: str,
    outcome: str,
    request_id: str | None,
    details: dict[str, Any],
    evidence: list[dict[str, Any]] | None = None,
) -> None:
    """写入轮次、证据和审计；原始请求文本不进入审计详情。"""

    safe_details = {
        key: value
        for key, value in details.items()
        if key not in {"text", "query", "content"}
    }
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO voice_turns(turn_id, session_id, user_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id, turn_id) DO UPDATE SET updated_at = NOW()
                WHERE voice_turns.user_id = EXCLUDED.user_id
                """,
                (turn_id, session_id, user_id),
            )
            if cur.rowcount != 1:
                raise PermissionError("turn is already bound to another user or session")
            for item in evidence or []:
                evidence_id = str(item.get("evidence_id") or "")
                if not evidence_id:
                    continue
                cur.execute(
                    """
                    INSERT INTO evidence(
                        evidence_id, turn_id, session_id, user_id, source_type,
                        source_id, document_id, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (evidence_id) DO NOTHING
                    """,
                    (
                        evidence_id,
                        turn_id,
                        session_id,
                        user_id,
                        str(item.get("source_type") or "unknown"),
                        str(item.get("source_id") or ""),
                        str(item.get("document_id") or ""),
                        json.dumps(item, ensure_ascii=False, default=str),
                    ),
                )
            cur.execute(
                """
                INSERT INTO audit_events(
                    user_id, session_id, turn_id, actor_type, action, outcome,
                    request_id, details
                ) VALUES (%s, %s, %s, 'worker', %s, %s, %s, %s)
                """,
                (
                    user_id,
                    session_id,
                    turn_id,
                    action,
                    outcome,
                    request_id,
                    json.dumps(safe_details, ensure_ascii=False, default=str),
                ),
            )


def record_voice_turn(
    *,
    user_id: str,
    session_id: str,
    turn_id: str,
    turn_index: int | None,
    user_text: str,
    raw_model_text: str,
    final_text: str,
    safety_result: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> None:
    """幂等写入单轮消息、安全结果和证据，全部受 turn_id 约束。"""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO voice_turns(
                    turn_id, session_id, user_id, turn_index, user_text,
                    assistant_text, raw_model_text, final_tts_text, safety_result
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id, turn_id) DO UPDATE SET
                    turn_index = COALESCE(EXCLUDED.turn_index, voice_turns.turn_index),
                    user_text = EXCLUDED.user_text,
                    assistant_text = EXCLUDED.assistant_text,
                    raw_model_text = EXCLUDED.raw_model_text,
                    final_tts_text = EXCLUDED.final_tts_text,
                    safety_result = EXCLUDED.safety_result,
                    updated_at = NOW()
                WHERE voice_turns.user_id = EXCLUDED.user_id
                """,
                (
                    turn_id,
                    session_id,
                    user_id,
                    turn_index,
                    user_text,
                    final_text,
                    raw_model_text,
                    final_text,
                    json.dumps(safety_result, ensure_ascii=False, default=str),
                ),
            )
            if cur.rowcount != 1:
                raise PermissionError("turn is already bound to another user or session")
            _upsert_conversation_message(
                cur,
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                role="user",
                content=user_text,
                metadata={"turn_index": turn_index},
            )
            _upsert_conversation_message(
                cur,
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                role="assistant",
                content=final_text,
                metadata={"turn_index": turn_index, "safety_checked": True},
            )
            for item in evidence:
                evidence_id = str(item.get("evidence_id") or "")
                if not evidence_id or str(item.get("turn_id") or "") != turn_id:
                    continue
                cur.execute(
                    """
                    INSERT INTO evidence(
                        evidence_id, turn_id, session_id, user_id, source_type,
                        source_id, document_id, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (evidence_id) DO UPDATE SET payload = EXCLUDED.payload
                    WHERE evidence.session_id = EXCLUDED.session_id
                      AND evidence.turn_id = EXCLUDED.turn_id
                      AND evidence.user_id = EXCLUDED.user_id
                    """,
                    (
                        evidence_id,
                        turn_id,
                        session_id,
                        user_id,
                        str(item.get("source_type") or "unknown"),
                        str(item.get("source_id") or ""),
                        str(item.get("document_id") or ""),
                        json.dumps(item, ensure_ascii=False, default=str),
                    ),
                )


def _upsert_conversation_message(
    cur: Any,
    *,
    user_id: str,
    session_id: str,
    turn_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any],
    source_type: str = "voice",
) -> None:
    """Write one canonical message; an empty side of a partial turn is omitted."""

    if not content.strip():
        return
    message_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"medagent:{session_id}:{turn_id}:{role}")
    )
    cur.execute(
        """
        INSERT INTO conversation_messages(
            message_id, user_id, session_id, turn_id, role, content,
            source_type, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id, turn_id, role) DO UPDATE SET
            content = EXCLUDED.content,
            source_type = EXCLUDED.source_type,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        WHERE conversation_messages.user_id = EXCLUDED.user_id
        """,
        (
            message_id,
            user_id,
            session_id,
            turn_id,
            role,
            content,
            source_type,
            json.dumps(metadata, ensure_ascii=False, default=str),
        ),
    )
    if cur.rowcount != 1:
        raise PermissionError("message is already bound to another user")


def record_text_message(
    *,
    user_id: str,
    username: str,
    session_id: str,
    turn_id: str,
    role: str,
    content: str,
    rag_trace: dict[str, Any] | None = None,
) -> None:
    """Atomically write one legacy-compatible and canonical text message."""

    with get_conn() as conn:
        with conn.cursor() as cur:
            _lock_or_create_text_session(
                cur,
                user_id=user_id,
                session_id=session_id,
                username=username,
            )
            _upsert_legacy_text_message(
                cur,
                session_id=session_id,
                turn_id=turn_id,
                role=role,
                content=content,
                rag_trace=rag_trace,
            )
            _upsert_conversation_message(
                cur,
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                role=role,
                content=content,
                metadata={},
                source_type="text",
            )
            _refresh_text_session_message_count(
                cur, session_id=session_id, username=username
            )


def record_text_turn(
    *,
    user_id: str,
    username: str,
    session_id: str,
    turn_id: str,
    user_text: str,
    assistant_text: str,
    rag_trace: dict[str, Any] | None = None,
) -> None:
    """Atomically persist both sides of a completed text turn."""

    with get_conn() as conn:
        with conn.cursor() as cur:
            _lock_or_create_text_session(
                cur,
                user_id=user_id,
                session_id=session_id,
                username=username,
            )
            for role, content, trace in (
                ("user", user_text, None),
                ("assistant", assistant_text, rag_trace),
            ):
                _upsert_legacy_text_message(
                    cur,
                    session_id=session_id,
                    turn_id=turn_id,
                    role=role,
                    content=content,
                    rag_trace=trace,
                )
                _upsert_conversation_message(
                    cur,
                    user_id=user_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    role=role,
                    content=content,
                    metadata={},
                    source_type="text",
                )
            _refresh_text_session_message_count(
                cur, session_id=session_id, username=username
            )


def _lock_or_create_text_session(
    cur: Any, *, user_id: str, session_id: str, username: str
) -> None:
    cur.execute(
        """
        INSERT INTO chat_sessions(session_id, username)
        VALUES (%s, %s)
        ON CONFLICT (session_id) DO NOTHING
        """,
        (session_id, username),
    )
    cur.execute(
        """
        SELECT ended_at FROM chat_sessions
        WHERE session_id = %s AND username = %s
        FOR UPDATE
        """,
        (session_id, username),
    )
    session = cur.fetchone()
    if session is None:
        raise PermissionError("session belongs to another user")
    ended_at = session.get("ended_at") if isinstance(session, dict) else session[0]
    if ended_at is not None:
        raise ValueError("text session is already finalized")
    cur.execute(
        """
        INSERT INTO conversation_sessions(
            session_type, session_id, user_id
        ) VALUES ('text', %s, %s)
        ON CONFLICT (session_type, session_id) DO UPDATE SET
            user_id = conversation_sessions.user_id
        WHERE conversation_sessions.user_id = EXCLUDED.user_id
        """,
        (session_id, user_id),
    )
    if cur.rowcount != 1:
        raise PermissionError("unified session belongs to another user")


def _upsert_legacy_text_message(
    cur: Any,
    *,
    session_id: str,
    turn_id: str,
    role: str,
    content: str,
    rag_trace: dict[str, Any] | None,
) -> None:
    if not content.strip():
        return
    msg_type = "human" if role == "user" else "ai"
    cur.execute(
        """
        INSERT INTO session_messages(
            session_id, turn_id, msg_type, content, rag_trace
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (session_id, turn_id, msg_type)
            WHERE turn_id IS NOT NULL
        DO UPDATE SET content = EXCLUDED.content, rag_trace = EXCLUDED.rag_trace
        """,
        (
            session_id,
            turn_id,
            msg_type,
            content,
            json.dumps(rag_trace, ensure_ascii=False, default=str)
            if rag_trace
            else None,
        ),
    )


def _refresh_text_session_message_count(
    cur: Any, *, session_id: str, username: str
) -> None:
    cur.execute(
        """
        UPDATE chat_sessions
        SET message_count = (
                SELECT COUNT(*) FROM session_messages WHERE session_id = %s
            ),
            updated_at = NOW()
        WHERE session_id = %s AND username = %s
        """,
        (session_id, session_id, username),
    )
    if cur.rowcount != 1:
        raise PermissionError("session belongs to another user")


def finalize_voice_session_memory(
    *, user_id: str, session_id: str, summary_version: int
) -> dict[str, Any]:
    """Idempotently close a voice PostgreSQL episodic/fact-memory loop."""

    return _finalize_session_memory(
        user_id=user_id,
        session_id=session_id,
        summary_version=summary_version,
        session_type="voice",
    )


def finalize_text_session_memory(
    *, user_id: str, session_id: str, summary_version: int
) -> dict[str, Any]:
    """Idempotently close a text PostgreSQL episodic/fact-memory loop."""

    return _finalize_session_memory(
        user_id=user_id,
        session_id=session_id,
        summary_version=summary_version,
        session_type="text",
    )


def _finalize_session_memory(
    *,
    user_id: str,
    session_id: str,
    summary_version: int,
    session_type: str,
) -> dict[str, Any]:
    """Shared transactional finalizer for voice and text sessions."""

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if session_type == "voice":
                cur.execute(
                    """
                    SELECT session_id FROM voice_sessions
                    WHERE session_id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (session_id, user_id),
                )
            elif session_type == "text":
                cur.execute(
                    """
                    SELECT sessions.session_id
                    FROM chat_sessions AS sessions
                    JOIN users ON users.username = sessions.username
                    WHERE sessions.session_id = %s AND users.user_id = %s
                    FOR UPDATE OF sessions
                    """,
                    (session_id, user_id),
                )
            else:
                raise ValueError("invalid session type")
            if cur.fetchone() is None:
                raise PermissionError("session does not belong to user")
            cur.execute(
                """
                SELECT * FROM session_summaries
                WHERE session_id = %s AND summary_version = %s
                """,
                (session_id, summary_version),
            )
            existing = cur.fetchone()
            if existing is not None:
                _mark_text_session_ended(
                    cur, session_id=session_id, session_type=session_type
                )
                return _finalize_result(cur, dict(existing), created=False)

            cur.execute(
                """
                SELECT turn_id, role, content
                FROM conversation_messages
                WHERE session_id = %s AND user_id = %s
                ORDER BY created_at, turn_id, role DESC
                """,
                (session_id, user_id),
            )
            messages = [dict(row) for row in cur.fetchall()]
            summary = build_session_summary(messages)
            summary_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"medagent:{session_id}:summary:{summary_version}",
                )
            )
            cur.execute(
                """
                INSERT INTO session_summaries(
                    summary_id, session_type, session_id, user_id,
                    summary_version, content,
                    structured_summary, source_digest, message_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    summary_id,
                    session_type,
                    session_id,
                    user_id,
                    summary_version,
                    summary["content"],
                    json.dumps(summary["structured_summary"], ensure_ascii=False),
                    summary["source_digest"],
                    summary["message_count"],
                ),
            )
            summary_row = dict(cur.fetchone())
            candidate_count = 0
            for message in messages:
                if message["role"] != "user":
                    continue
                candidate_count += _insert_candidates(
                    cur,
                    user_id=user_id,
                    session_id=session_id,
                    source_type="user_message",
                    session_type=session_type,
                    source_id=str(message["turn_id"]),
                    source_turn_id=str(message["turn_id"]),
                    source_document_id=None,
                    text=str(message["content"]),
                    confirmed=False,
                )
            cur.execute(
                """
                SELECT payload FROM evidence
                WHERE session_id = %s AND user_id = %s AND source_type = 'personal'
                """,
                (session_id, user_id),
            )
            seen_documents: set[str] = set()
            for row in cur.fetchall():
                payload = row["payload"]
                if not isinstance(payload, dict):
                    continue
                document_id = str(payload.get("document_id") or "")
                if not document_id or document_id in seen_documents:
                    continue
                preview = str(payload.get("content_preview") or "").strip()
                if not preview:
                    continue
                seen_documents.add(document_id)
                server_verified = _is_server_verified_personal_document_content(
                    cur,
                    user_id=user_id,
                    document_id=document_id,
                    content=preview,
                )
                candidate_count += _insert_candidates(
                    cur,
                    user_id=user_id,
                    session_id=session_id,
                    source_type="personal_document",
                    session_type=session_type,
                    source_id=document_id,
                    source_turn_id=None,
                    source_document_id=document_id,
                    text=preview,
                    confirmed=server_verified,
                )
            _mark_text_session_ended(
                cur, session_id=session_id, session_type=session_type
            )
            result = _finalize_result(cur, summary_row, created=True)
            result["created_candidates"] = candidate_count
            return result


def _mark_text_session_ended(
    cur: Any, *, session_id: str, session_type: str
) -> None:
    if session_type != "text":
        return
    cur.execute(
        """
        UPDATE chat_sessions
        SET ended_at = COALESCE(ended_at, NOW()), updated_at = NOW()
        WHERE session_id = %s
        """,
        (session_id,),
    )
    cur.execute(
        """
        UPDATE conversation_sessions
        SET ended_at = COALESCE(ended_at, NOW())
        WHERE session_type = 'text' AND session_id = %s
        """,
        (session_id,),
    )


def _is_server_verified_personal_document_content(
    cur: Any, *, user_id: str, document_id: str, content: str
) -> bool:
    """Trust only an exact content digest registered by server-side processing."""

    normalized = " ".join(unicodedata.normalize("NFKC", content).split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    cur.execute(
        """
        SELECT 1 FROM trusted_personal_document_contents
        WHERE user_id = %s AND document_id = %s AND content_sha256 = %s
          AND status = 'trusted' AND revoked_at IS NULL
        """,
        (user_id, document_id, digest),
    )
    return cur.fetchone() is not None


def _insert_candidates(
    cur: Any,
    *,
    user_id: str,
    session_id: str,
    session_type: str,
    source_type: str,
    source_id: str,
    source_turn_id: str | None,
    source_document_id: str | None,
    text: str,
    confirmed: bool,
) -> int:
    inserted = 0
    for candidate in extract_medical_fact_candidates(text, source_type=source_type):
        key = candidate_key(
            source_type=source_type,
            source_id=source_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
        )
        cur.execute(
            """
            INSERT INTO medical_fact_memories(
                memory_id, user_id, memory_type, content, structured_value,
                status, source_type, source_session_type, source_session_id,
                source_turn_id, source_document_id, confidence, candidate_key
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, candidate_key) DO NOTHING
            """,
            (
                str(uuid.uuid4()),
                user_id,
                candidate.memory_type,
                candidate.content,
                json.dumps(candidate.structured_value, ensure_ascii=False),
                "confirmed" if confirmed else "proposed",
                source_type,
                session_type,
                session_id,
                source_turn_id,
                source_document_id,
                candidate.confidence,
                key,
            ),
        )
        inserted += max(cur.rowcount, 0)
    return inserted


def _finalize_result(
    cur: Any, summary: dict[str, Any], *, created: bool
) -> dict[str, Any]:
    cur.execute(
        """
        SELECT COUNT(*) FROM medical_fact_memories
        WHERE source_session_type = %s AND source_session_id = %s
          AND deleted_at IS NULL
        """,
        (summary["session_type"], summary["session_id"]),
    )
    count = int(cur.fetchone()["count"])
    return {
        "session_id": summary["session_id"],
        "summary_id": str(summary["summary_id"]),
        "summary_version": int(summary["summary_version"]),
        "message_count": int(summary["message_count"]),
        "memory_candidate_count": count,
        "created": created,
        "replacement_verified": True,
    }


def list_medical_fact_memories(
    *, user_id: str, statuses: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = """
                SELECT * FROM medical_fact_memories
                WHERE user_id = %s AND deleted_at IS NULL
            """
            params: list[Any] = [user_id]
            if statuses:
                query += " AND status = ANY(%s)"
                params.append(list(statuses))
            query += " ORDER BY created_at DESC, memory_id"
            cur.execute(query, params)
            return [_memory_row(row) for row in cur.fetchall()]


def set_medical_fact_status(
    *, user_id: str, memory_id: str, status: str
) -> dict[str, Any] | None:
    if status not in {"confirmed", "rejected"}:
        raise ValueError("invalid target memory status")
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE medical_fact_memories
                SET status = %s, updated_at = NOW()
                WHERE memory_id = %s AND user_id = %s AND deleted_at IS NULL
                  AND status IN ('proposed', %s)
                RETURNING *
                """,
                (status, memory_id, user_id, status),
            )
            row = cur.fetchone()
            return _memory_row(row) if row else None


def correct_medical_fact_memory(
    *,
    user_id: str,
    memory_id: str,
    content: str,
    structured_value: dict[str, Any],
    memory_type: str | None = None,
    confidence: float = 1.0,
) -> dict[str, Any] | None:
    """Create a confirmed replacement and supersede the old version atomically."""

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM medical_fact_memories
                WHERE memory_id = %s AND user_id = %s AND deleted_at IS NULL
                FOR UPDATE
                """,
                (memory_id, user_id),
            )
            old = cur.fetchone()
            if old is None:
                return None
            replacement_type = memory_type or str(old["memory_type"])
            key = candidate_key(
                source_type="user_correction",
                source_id=memory_id,
                memory_type=replacement_type,
                content=content,
            )
            cur.execute(
                """
                SELECT * FROM medical_fact_memories
                WHERE user_id = %s AND candidate_key = %s AND deleted_at IS NULL
                """,
                (user_id, key),
            )
            existing = cur.fetchone()
            if existing is not None:
                return _memory_row(existing)
            if old["status"] not in {"proposed", "confirmed"}:
                raise ValueError("only proposed or confirmed memory can be corrected")
            new_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO medical_fact_memories(
                    memory_id, user_id, memory_type, content, structured_value,
                    status, source_type, source_session_type,
                    source_session_id, confidence, supersedes_memory_id,
                    candidate_key
                ) VALUES (%s, %s, %s, %s, %s, 'confirmed', 'user_correction',
                          %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    new_id,
                    user_id,
                    replacement_type,
                    content,
                    json.dumps(structured_value, ensure_ascii=False),
                    old["source_session_type"],
                    old["source_session_id"],
                    confidence,
                    memory_id,
                    key,
                ),
            )
            replacement = dict(cur.fetchone())
            cur.execute(
                """
                UPDATE medical_fact_memories
                SET status = 'superseded', updated_at = NOW()
                WHERE memory_id = %s AND user_id = %s
                """,
                (memory_id, user_id),
            )
            return _memory_row(replacement)


def delete_medical_fact_memory(*, user_id: str, memory_id: str) -> bool:
    """Irreversibly redact content while retaining a minimal audit identity."""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE medical_fact_memories
                SET content = '', structured_value = '{}'::jsonb,
                    deleted_at = COALESCE(deleted_at, NOW()), updated_at = NOW()
                WHERE memory_id = %s AND user_id = %s
                """,
                (memory_id, user_id),
            )
            return cur.rowcount == 1


def export_controlled_memory(*, user_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM medical_fact_memories
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY created_at, memory_id
                """,
                (user_id,),
            )
            memories = [_memory_row(row) for row in cur.fetchall()]
            cur.execute(
                """
                SELECT * FROM session_summaries
                WHERE user_id = %s ORDER BY created_at, summary_id
                """,
                (user_id,),
            )
            summaries = [_json_row(row) for row in cur.fetchall()]
    return {
        "schema_version": 1,
        "user_id": user_id,
        "medical_fact_memories": memories,
        "session_summaries": summaries,
    }


def _memory_row(row: Any) -> dict[str, Any]:
    return _json_row(row)


def _json_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key, value in list(data.items()):
        if isinstance(value, uuid.UUID):
            data[key] = str(value)
        elif isinstance(value, datetime):
            data[key] = value.isoformat()
    return data


def request_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
