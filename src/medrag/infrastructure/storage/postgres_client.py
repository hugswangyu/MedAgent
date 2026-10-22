"""PostgreSQL 存储后端。

替代 JSON 文件持久化，提供会话、LTM 记忆和用户偏好的读写。
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Generator, List, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from medrag.config.settings import settings

logger = logging.getLogger(__name__)

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=settings.pg_host,
            port=settings.pg_port,
            user=settings.pg_user,
            password=settings.pg_password,
            dbname=settings.pg_database,
        )
    return _pool


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    conn = get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)


# ---------------------------------------------------------------------------
# 长期记忆 (LTM)
# ---------------------------------------------------------------------------


def ltm_save_item(
    username: str,
    content: str,
    importance: float,
    embedding: Optional[List[float]],
    category: str,
    tags: List[str],
    slot_hint: str,
    created_at: Optional[str] = None,
) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO long_term_memory
                (username, content, importance, embedding, category, tags, slot_hint, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                username,
                content,
                importance,
                embedding,
                category,
                tags,
                slot_hint,
                created_at or datetime.now().isoformat(),
            ),
        )
        return cur.fetchone()[0]


def ltm_update_item(
    item_id: int,
    content: Optional[str] = None,
    importance: Optional[float] = None,
    embedding: Optional[List[float]] = None,
    tags: Optional[List[str]] = None,
    last_accessed: Optional[str] = None,
) -> None:
    fields = []
    values = []
    if content is not None:
        fields.append("content = %s")
        values.append(content)
    if importance is not None:
        fields.append("importance = %s")
        values.append(importance)
    if embedding is not None:
        fields.append("embedding = %s")
        values.append(embedding)
    if tags is not None:
        fields.append("tags = %s")
        values.append(tags)
    if last_accessed is not None:
        fields.append("last_accessed = %s")
        values.append(last_accessed)
    if not fields:
        return
    values.append(item_id)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE long_term_memory SET " + ", ".join(fields) + " WHERE id = %s",
            values,
        )


def ltm_delete_items(ids: List[int]) -> None:
    if not ids:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM long_term_memory WHERE id = ANY(%s)", (ids,)
        )


def ltm_load_all(username: str) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM long_term_memory WHERE username = %s ORDER BY id",
            (username,),
        )
        return [dict(r) for r in cur.fetchall()]


def ltm_sync_items(username: str, rows: List[Dict], deleted_ids: List[int]) -> List[int]:
    """Atomically persist revisions, decay and deletions with tenant-scoped writes.

    Requires scripts/migrations/2026-10-21-atomic-memory.sql on existing databases.
    A row with id=None is a new item; other rows must already belong to this user.
    """
    assigned = []
    with get_conn() as conn:
        cur = conn.cursor()
        if deleted_ids:
            cur.execute("DELETE FROM long_term_memory WHERE username = %s AND id = ANY(%s)",
                        (username, deleted_ids))
        for row in rows:
            values = (row["content"], row["importance"], row.get("embedding"),
                      row.get("category", "general"), row.get("tags", []), row.get("slot_hint", ""),
                      row.get("created_at"), row.get("last_accessed"),
                      row.get("base_importance"), row.get("decay_anchor"),
                      psycopg2.extras.Json(row.get("fact")))
            if row.get("id") is None:
                cur.execute("""INSERT INTO long_term_memory
                    (content, importance, embedding, category, tags, slot_hint, created_at,
                     last_accessed, base_importance, decay_anchor, fact, username)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                            values + (username,))
            else:
                cur.execute("""UPDATE long_term_memory SET
                    content=%s, importance=%s, embedding=%s, category=%s, tags=%s, slot_hint=%s,
                    created_at=%s, last_accessed=%s, base_importance=%s, decay_anchor=%s, fact=%s
                    WHERE username=%s AND id=%s RETURNING id""", values + (username, row["id"]))
            result = cur.fetchone()
            if result is None:
                raise ValueError("Memory row missing or belongs to a different user")
            assigned.append(result[0])
    return assigned


def pref_clear(username: str) -> None:
    with get_conn() as conn:
        conn.cursor().execute("DELETE FROM user_preferences WHERE username = %s", (username,))


def ltm_recall(username: str, category_filter: Optional[List[str]] = None) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if category_filter:
            cur.execute(
                "SELECT * FROM long_term_memory WHERE username = %s AND category = ANY(%s) ORDER BY importance DESC, last_accessed DESC",
                (username, category_filter),
            )
        else:
            cur.execute(
                "SELECT * FROM long_term_memory WHERE username = %s ORDER BY importance DESC, last_accessed DESC",
                (username,),
            )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# 会话 (Sessions)
# ---------------------------------------------------------------------------


def session_save(
    session_id: str, username: str, title: str = "新对话"
) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_sessions (session_id, username, title)
            VALUES (%s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (session_id, username, title),
        )


def session_update(session_id: str, message_count: int) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE chat_sessions SET message_count = %s, updated_at = NOW() WHERE session_id = %s",
            (message_count, session_id),
        )


def session_list(username: str) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT session_id, title, message_count, updated_at
            FROM chat_sessions
            WHERE username = %s
            ORDER BY updated_at DESC
            """,
            (username,),
        )
        return [dict(r) for r in cur.fetchall()]


def session_get(session_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM chat_sessions WHERE session_id = %s", (session_id,)
        )
        r = cur.fetchone()
        return dict(r) if r else None


def session_delete(session_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))


def message_add(
    session_id: str, msg_type: str, content: str, rag_trace: Optional[Dict] = None
) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO session_messages (session_id, msg_type, content, rag_trace)
            VALUES (%s, %s, %s, %s)
            """,
            (session_id, msg_type, content, json.dumps(rag_trace) if rag_trace else None),
        )


def message_list(session_id: str) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT msg_type, content, rag_trace, created_at
            FROM session_messages
            WHERE session_id = %s
            ORDER BY id
            """,
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# 用户偏好 (Preferences)
# ---------------------------------------------------------------------------


def pref_save(username: str, key: str, value: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_preferences (username, key, value)
            VALUES (%s, %s, %s)
            ON CONFLICT (username, key) DO UPDATE SET value = EXCLUDED.value
            """,
            (username, key, value),
        )


def pref_load_all(username: str) -> Dict[str, str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT key, value FROM user_preferences WHERE username = %s",
            (username,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}
