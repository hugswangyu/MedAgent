"""Real PostgreSQL acceptance for the Phase 3 unified session migration."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import psycopg2
import psycopg2.extras
import pytest


def _dsn() -> str:
    value = os.getenv("MEDAGENT_TEST_POSTGRES_DSN", "").strip()
    if not value:
        pytest.skip("MEDAGENT_TEST_POSTGRES_DSN is required")
    return value


@contextmanager
def _schema_connection(dsn: str, schema: str):
    connection = psycopg2.connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema}"')
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _legacy_schema(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE users (
            user_id UUID PRIMARY KEY,
            username TEXT NOT NULL UNIQUE
        );
        CREATE TABLE chat_sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '新对话',
            message_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE session_messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES chat_sessions(session_id)
                ON DELETE CASCADE,
            msg_type TEXT NOT NULL,
            content TEXT NOT NULL,
            rag_trace JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE voice_sessions (
            session_id TEXT PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(user_id),
            knowledge_base_id TEXT NOT NULL DEFAULT 'kb-test',
            status TEXT NOT NULL DEFAULT 'created',
            room_name TEXT,
            binding_version BIGINT NOT NULL DEFAULT 1,
            livekit_job_id TEXT,
            client_id TEXT,
            client_type TEXT NOT NULL DEFAULT 'web',
            participant_identity TEXT,
            token_expires_at TIMESTAMPTZ,
            lease_expires_at TIMESTAMPTZ,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_at TIMESTAMPTZ
        );
        CREATE TABLE evidence (
            evidence_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id UUID NOT NULL,
            source_type TEXT NOT NULL,
            payload JSONB NOT NULL
        );
        CREATE TABLE session_summaries (
            summary_id UUID PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES voice_sessions(session_id)
                ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(user_id),
            summary_version INTEGER NOT NULL,
            content TEXT NOT NULL,
            structured_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_digest TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(session_id, summary_version)
        );
        CREATE TABLE medical_fact_memories (
            memory_id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(user_id),
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            structured_value JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_session_id TEXT REFERENCES voice_sessions(session_id)
                ON DELETE SET NULL,
            source_turn_id TEXT,
            source_document_id TEXT,
            valid_from TIMESTAMPTZ,
            valid_to TIMESTAMPTZ,
            confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            supersedes_memory_id UUID REFERENCES medical_fact_memories(memory_id),
            candidate_key TEXT NOT NULL,
            deleted_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(user_id, candidate_key)
        );
        """
    )


def test_real_postgres_migration_and_text_finalize(monkeypatch):
    from medrag.infrastructure.storage import phase1_repository

    dsn = _dsn()
    schema = f"phase3_{uuid.uuid4().hex}"
    user_id = str(uuid.uuid4())
    voice_summary_id = str(uuid.uuid4())
    voice_memory_id = str(uuid.uuid4())
    migration = (
        Path(__file__).resolve().parents[1] / "scripts" / "phase3_memory.sql"
    ).read_text(encoding="utf-8")

    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA "{schema}"')

        with _schema_connection(dsn, schema) as connection:
            with connection.cursor() as cursor:
                _legacy_schema(cursor)
                cursor.execute(
                    "INSERT INTO users(user_id, username) VALUES (%s, 'alice')",
                    (user_id,),
                )
                cursor.execute(
                    "INSERT INTO voice_sessions(session_id, user_id) "
                    "VALUES ('voice-existing', %s)",
                    (user_id,),
                )
                cursor.execute(
                    "INSERT INTO chat_sessions(session_id, username) "
                    "VALUES ('text-existing', 'alice')"
                )
                cursor.execute(
                    """
                    INSERT INTO session_summaries(
                        summary_id, session_id, user_id, summary_version,
                        content, source_digest
                    ) VALUES (%s, 'voice-existing', %s, 1, 'legacy', 'digest')
                    """,
                    (voice_summary_id, user_id),
                )
                cursor.execute(
                    """
                    INSERT INTO medical_fact_memories(
                        memory_id, user_id, memory_type, content, status,
                        source_type, source_session_id, confidence, candidate_key
                    ) VALUES (
                        %s, %s, 'allergy', '过敏：青霉素', 'proposed',
                        'user_message', 'voice-existing', 0.75, 'legacy-key'
                    )
                    """,
                    (voice_memory_id, user_id),
                )
                cursor.execute(migration)
                cursor.execute(migration)

        @contextmanager
        def test_get_conn():
            with _schema_connection(dsn, schema) as connection:
                yield connection

        monkeypatch.setattr(phase1_repository, "get_conn", test_get_conn)

        phase1_repository.record_text_turn(
            user_id=user_id,
            username="alice",
            session_id="text-existing",
            turn_id="text-turn-1",
            user_text="我对头孢过敏",
            assistant_text="请确认这条过敏信息。",
        )
        first = phase1_repository.finalize_text_session_memory(
            user_id=user_id,
            session_id="text-existing",
            summary_version=1,
        )
        second = phase1_repository.finalize_text_session_memory(
            user_id=user_id,
            session_id="text-existing",
            summary_version=1,
        )

        assert first["created"] is True
        assert second["created"] is False

        voice = phase1_repository.create_voice_session_binding(
            user_id=user_id,
            session_id="voice-new",
            knowledge_base_id="kb-test",
            room_name="room-voice-new",
        )
        assert voice["user_id"] == user_id

        with _schema_connection(dsn, schema) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_messages(
                        message_id, user_id, session_id, turn_id, role,
                        content, source_type
                    ) VALUES
                        (%s, %s, 'voice-new', 'voice-turn-1', 'user',
                         '我正在服用二甲双胍', 'voice'),
                        (%s, %s, 'voice-new', 'voice-turn-1', 'assistant',
                         '请确认当前用药。', 'voice')
                    """,
                    (str(uuid.uuid4()), user_id, str(uuid.uuid4()), user_id),
                )

        voice_first = phase1_repository.finalize_voice_session_memory(
            user_id=user_id,
            session_id="voice-new",
            summary_version=1,
        )
        voice_second = phase1_repository.finalize_voice_session_memory(
            user_id=user_id,
            session_id="voice-new",
            summary_version=1,
        )
        assert voice_first["created"] is True
        assert voice_second["created"] is False

        other_user_id = str(uuid.uuid4())
        with _schema_connection(dsn, schema) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users(user_id, username) VALUES (%s, 'bob')",
                    (other_user_id,),
                )
                cursor.execute(
                    """
                    INSERT INTO conversation_sessions(
                        session_type, session_id, user_id
                    ) VALUES ('voice', 'voice-conflict', %s)
                    """,
                    (other_user_id,),
                )

        with pytest.raises(PermissionError, match="another user"):
            phase1_repository.create_voice_session_binding(
                user_id=user_id,
                session_id="voice-conflict",
                knowledge_base_id="kb-test",
                room_name="room-voice-conflict",
            )

        with _schema_connection(dsn, schema) as connection:
            with connection.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cursor:
                cursor.execute(
                    """
                    SELECT session_type, session_id FROM session_summaries
                    ORDER BY session_type, session_id
                    """
                )
                summaries = [dict(row) for row in cursor.fetchall()]
                assert summaries == [
                    {"session_type": "text", "session_id": "text-existing"},
                    {"session_type": "voice", "session_id": "voice-existing"},
                    {"session_type": "voice", "session_id": "voice-new"},
                ]

                cursor.execute(
                    """
                    SELECT source_session_type, source_session_id, status
                    FROM medical_fact_memories
                    ORDER BY source_session_type, source_session_id
                    """
                )
                memories = [dict(row) for row in cursor.fetchall()]
                assert memories == [
                    {
                        "source_session_type": "text",
                        "source_session_id": "text-existing",
                        "status": "proposed",
                    },
                    {
                        "source_session_type": "voice",
                        "source_session_id": "voice-existing",
                        "status": "proposed",
                    },
                    {
                        "source_session_type": "voice",
                        "source_session_id": "voice-new",
                        "status": "proposed",
                    },
                ]

                cursor.execute(
                    "SELECT COUNT(*) AS count FROM voice_sessions "
                    "WHERE session_id = 'voice-conflict'"
                )
                assert cursor.fetchone()["count"] == 0

                cursor.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM pg_constraint
                    WHERE conname IN (
                        'session_summaries_session_id_fkey',
                        'medical_fact_memories_source_session_id_fkey'
                    )
                    """
                )
                assert cursor.fetchone()["count"] == 0
    finally:
        with admin.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.close()
