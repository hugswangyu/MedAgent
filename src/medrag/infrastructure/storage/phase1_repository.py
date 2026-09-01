"""阶段 1 身份、语音会话与审计 PostgreSQL 仓储。"""

from __future__ import annotations

import hashlib
import json
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
    password_hash: str
    is_admin: bool


def ensure_schema() -> None:
    """幂等安装阶段 1 schema。"""

    migration = Path(__file__).resolve().parents[4] / "scripts" / "phase1_identity_data.sql"
    sql = migration.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def import_legacy_users(users: Iterable[tuple[str, str, bool]]) -> int:
    """只导入缺失的旧用户；绝不改写已有 PostgreSQL 身份。"""

    imported = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for username, password_hash, is_admin in users:
                cur.execute(
                    """
                    INSERT INTO users(user_id, username, password_hash, is_admin)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (username) DO NOTHING
                    """,
                    (str(uuid.uuid4()), username, password_hash, is_admin),
                )
                imported += max(cur.rowcount, 0)
    return imported


def get_user_by_username(username: str) -> UserRecord | None:
    return _get_user("username = %s", username)


def get_user_by_id(user_id: str) -> UserRecord | None:
    return _get_user("user_id = %s", user_id)


def _get_user(predicate: str, value: str) -> UserRecord | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT user_id, username, password_hash, is_admin FROM users WHERE {predicate}",
                (value,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return UserRecord(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        password_hash=str(row["password_hash"]),
        is_admin=bool(row["is_admin"]),
    )


def create_user(username: str, password_hash: str, is_admin: bool) -> UserRecord | None:
    user_id = str(uuid.uuid4())
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO users(user_id, username, password_hash, is_admin)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (username) DO NOTHING
                RETURNING user_id, username, password_hash, is_admin
                """,
                (user_id, username, password_hash, is_admin),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return UserRecord(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        password_hash=str(row["password_hash"]),
        is_admin=bool(row["is_admin"]),
    )


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


def bind_voice_session(
    *,
    user_id: str,
    session_id: str,
    knowledge_base_id: str,
    room_name: str | None = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO voice_sessions(session_id, user_id, knowledge_base_id, room_name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    updated_at = NOW(),
                    room_name = COALESCE(EXCLUDED.room_name, voice_sessions.room_name)
                WHERE voice_sessions.user_id = EXCLUDED.user_id
                  AND voice_sessions.knowledge_base_id = EXCLUDED.knowledge_base_id
                """,
                (session_id, user_id, knowledge_base_id, room_name),
            )
            if cur.rowcount != 1:
                raise PermissionError(
                    "voice session is already bound to another user or knowledge base"
                )


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
