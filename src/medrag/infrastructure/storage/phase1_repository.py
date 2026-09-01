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
    """幂等安装阶段 1 schema。"""

    migration = Path(__file__).resolve().parents[4] / "scripts" / "phase1_identity_data.sql"
    sql = migration.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
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
) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO voice_sessions(
                    session_id, user_id, knowledge_base_id, room_name, client_id,
                    client_type, participant_identity, token_expires_at, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                ),
            )
            row = cur.fetchone()
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
) -> dict[str, Any] | None:
    """以 binding_version + 空 job 条件执行一次 LiveKit job CAS。"""

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE voice_sessions
                SET livekit_job_id = %s, status = 'active',
                    binding_version = binding_version + 1, updated_at = NOW()
                WHERE session_id = %s
                  AND binding_version = %s
                  AND room_name = %s
                  AND status = 'created'
                  AND livekit_job_id IS NULL
                RETURNING *
                """,
                (livekit_job_id, session_id, expected_version, room_name),
            )
            row = cur.fetchone()
    return _voice_binding(row) if row else None


def validate_claimed_worker_binding(
    *, session_id: str, user_id: str, knowledge_base_id: str
) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM voice_sessions
                WHERE session_id = %s AND user_id = %s AND knowledge_base_id = %s
                  AND status = 'active' AND livekit_job_id IS NOT NULL
                """,
                (session_id, user_id, knowledge_base_id),
            )
            return cur.fetchone() is not None


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
            return cur.rowcount == 1


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


def request_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
