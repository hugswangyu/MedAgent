"""Offline acceptance tests for the Phase 3 controlled-memory loop."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from medrag.contracts.phase0 import (
    VoiceMemoryContextRequest,
    VoiceSessionFinalizeRequest,
)
from medrag.memory.controlled import (
    build_session_summary,
    extract_medical_fact_candidates,
)


def test_phase3_migration_defines_canonical_messages_summary_and_memory_chain():
    migration = (
        Path(__file__).resolve().parents[1] / "scripts" / "phase3_memory.sql"
    ).read_text(encoding="utf-8")

    assert "conversation_messages" in migration
    assert "conversation_messages_session_id_fkey" in migration
    assert "UNIQUE(session_id, turn_id, role)" in migration
    assert "conversation_sessions" in migration
    assert "uq_session_summaries_unified_version" in migration
    assert "session_summaries_unified_session_fkey" in migration
    assert "medical_fact_memories_unified_session_fkey" in migration
    assert "REFERENCES voice_sessions(session_id)" not in migration
    assert "supersedes_memory_id" in migration
    assert "deleted_at" in migration
    assert "trusted_personal_document_contents" in migration
    assert "ADD COLUMN IF NOT EXISTS ended_at" in migration
    assert "uq_session_messages_canonical_turn" in migration


def test_worker_token_has_dedicated_minimal_memory_read_scope():
    from medrag.app.auth_manager import WORKER_SCOPES

    assert "voice:memory:read" in WORKER_SCOPES


def test_text_chat_uses_working_memory_without_legacy_fact_writes():
    from medrag.memory import MemorySystem
    from medrag.service.chat_service import MedicalChatService

    memory = MemorySystem(max_turns=5)
    MedicalChatService._record_working_message(
        memory, "assistant", "模型推测用户患有糖尿病"
    )

    assert memory.stats["stm_count"] == 1
    assert memory.stats["ltm_count"] == 0


def test_voice_turn_writes_canonical_messages_in_same_repository_transaction():
    from medrag.infrastructure.storage import phase1_repository

    source = inspect.getsource(phase1_repository.record_voice_turn)
    assert source.count("_upsert_conversation_message(") == 2
    assert 'role="user"' in source
    assert 'role="assistant"' in source


def test_unified_voice_session_registration_is_in_creation_not_user_lookup():
    from medrag.infrastructure.storage import phase1_repository

    user_lookup = inspect.getsource(phase1_repository._get_user)
    voice_creation = inspect.getsource(
        phase1_repository.create_voice_session_binding
    )

    assert "conversation_sessions" not in user_lookup
    assert "session_id" not in user_lookup
    assert voice_creation.count("INSERT INTO conversation_sessions") == 1
    assert "conversation_sessions.user_id = EXCLUDED.user_id" in voice_creation
    assert "unified voice session belongs to another user" in voice_creation


def test_text_message_write_uses_one_atomic_repository_call(monkeypatch):
    from medrag.app import session_store

    captured = {}
    monkeypatch.setattr(
        session_store.phase1_repository,
        "record_text_turn",
        lambda **kwargs: captured.update(kwargs),
    )

    session_store.add_turn(
        "chat-1",
        user_text="用户消息",
        assistant_text="助手消息",
        username="alice",
        user_id="00000000-0000-0000-0000-000000000001",
        turn_id="text-turn-1",
    )

    assert captured == {
        "user_id": "00000000-0000-0000-0000-000000000001",
        "username": "alice",
        "session_id": "chat-1",
        "turn_id": "text-turn-1",
        "user_text": "用户消息",
        "assistant_text": "助手消息",
        "rag_trace": None,
    }


def test_text_turn_legacy_and_canonical_rows_share_one_transaction(monkeypatch):
    from medrag.infrastructure.storage import phase1_repository

    queries = []
    connections = 0

    class FakeCursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params):
            queries.append(" ".join(query.split()))

        def fetchone(self):
            return {"ended_at": None}

    class FakeConnection:
        def cursor(self, **kwargs):
            return FakeCursor()

    @contextmanager
    def fake_get_conn():
        nonlocal connections
        connections += 1
        yield FakeConnection()

    monkeypatch.setattr(phase1_repository, "get_conn", fake_get_conn)

    phase1_repository.record_text_turn(
        user_id="user-1",
        username="alice",
        session_id="chat-1",
        turn_id="turn-1",
        user_text="我对青霉素过敏",
        assistant_text="已记录，请确认。",
    )

    assert connections == 1
    assert sum(query.startswith("INSERT INTO session_messages") for query in queries) == 2
    assert sum(query.startswith("INSERT INTO conversation_messages") for query in queries) == 2


def test_text_stream_reports_persistence_failure_instead_of_swallowing_it():
    from medrag.app.api import chat

    source = inspect.getsource(chat.chat_stream)
    assert "add_turn" in source
    assert "消息持久化失败，请重试" in source
    assert "保存会话消息失败" not in source


@pytest.mark.parametrize(
    "text",
    [
        "我没有青霉素过敏",
        "我以前对青霉素过敏但已经好了",
        "我妈妈有高血压",
        "高血压是什么？",
    ],
)
def test_unsafe_contexts_do_not_become_medical_fact_candidates(text):
    assert extract_medical_fact_candidates(text, source_type="user_message") == []


def test_only_user_or_trusted_document_text_is_extractable():
    user = extract_medical_fact_candidates(
        "我对青霉素过敏。我正在服用二甲双胍",
        source_type="user_message",
    )
    document = extract_medical_fact_candidates(
        "过敏史：磺胺类\n血糖：7.2 mmol/L",
        source_type="personal_document",
    )

    assert {(item.memory_type, item.content) for item in user} == {
        ("allergy", "过敏：青霉素"),
        ("medication", "用药：二甲双胍"),
    }
    assert {item.memory_type for item in document} == {"allergy", "measurement"}
    assert extract_medical_fact_candidates(
        "患者肯定患有糖尿病", source_type="assistant"
    ) == []


def test_personal_document_confirmation_requires_server_digest_registration():
    from medrag.infrastructure.storage import phase1_repository

    class FakeCursor:
        def __init__(self, registered):
            self.registered = registered
            self.params = None

        def execute(self, query, params):
            self.params = params

        def fetchone(self):
            return (1,) if self.registered else None

    content = "过敏史：青霉素"
    unregistered = FakeCursor(False)
    trusted = FakeCursor(True)

    assert not phase1_repository._is_server_verified_personal_document_content(
        unregistered,
        user_id="user-1",
        document_id="doc-1",
        content=content,
    )
    assert phase1_repository._is_server_verified_personal_document_content(
        trusted,
        user_id="user-1",
        document_id="doc-1",
        content=content,
    )
    assert unregistered.params == (
        "user-1",
        "doc-1",
        hashlib.sha256(
            " ".join(unicodedata.normalize("NFKC", content).split()).encode("utf-8")
        ).hexdigest(),
    )

    finalize_source = inspect.getsource(phase1_repository._finalize_session_memory)
    assert "verification_status" not in finalize_source
    assert "confirmed=server_verified" in finalize_source


def test_unregistered_personal_document_candidate_is_proposed(monkeypatch):
    from medrag.infrastructure.storage import phase1_repository

    statuses = []

    class FakeCursor:
        rowcount = 1

        def execute(self, query, params):
            statuses.append(params[5])

    inserted = phase1_repository._insert_candidates(
        FakeCursor(),
        user_id="user-1",
        session_id="session-1",
        session_type="text",
        source_type="personal_document",
        source_id="doc-1",
        source_turn_id=None,
        source_document_id="doc-1",
        text="过敏史：青霉素",
        confirmed=False,
    )

    assert inserted == 1
    assert statuses == ["proposed"]


def test_session_summary_is_extractive_and_digest_is_stable():
    messages = [
        {"turn_id": "turn_1", "role": "user", "content": " 我有高血压 "},
        {"turn_id": "turn_1", "role": "assistant", "content": "建议记录血压。"},
    ]

    first = build_session_summary(messages)
    second = build_session_summary(messages)

    assert first == second
    assert first["structured_summary"]["extractive"] is True
    assert "我有高血压" in first["content"]
    assert first["message_count"] == 2


def test_finalize_endpoint_is_idempotent_before_repository(monkeypatch):
    from medrag.app.api import internal_v1

    calls = []
    monkeypatch.setattr(
        internal_v1,
        "_authorize",
        lambda **kwargs: (None, SimpleNamespace(user_id="user_1")),
    )
    monkeypatch.setattr(
        internal_v1.phase1_repository,
        "finalize_voice_session_memory",
        lambda **kwargs: calls.append(kwargs)
        or {
            "session_id": "vs_1",
            "summary_id": "summary_1",
            "summary_version": 1,
            "replacement_verified": True,
        },
    )
    payload = VoiceSessionFinalizeRequest(
        session_id="vs_1",
        summary_version=1,
        idempotency_key="phase3-finalize-offline-1",
    )

    first = asyncio.run(
        internal_v1.finalize_voice_session(
            "vs_1",
            payload,
            authorization="Bearer test",
            x_request_nonce="00000000-0000-0000-0000-000000000001",
        )
    )
    second = asyncio.run(
        internal_v1.finalize_voice_session(
            "vs_1",
            payload,
            authorization="Bearer test",
            x_request_nonce="00000000-0000-0000-0000-000000000002",
        )
    )

    assert json.loads(first.body)["data"]["replacement_verified"] is True
    assert json.loads(second.body)["metrics"]["idempotency_replay"] is True
    assert calls == [
        {"user_id": "user_1", "session_id": "vs_1", "summary_version": 1}
    ]


def test_text_finalize_uses_shared_controlled_memory_loop(monkeypatch):
    from medrag.infrastructure.storage import phase1_repository

    captured = {}
    monkeypatch.setattr(
        phase1_repository,
        "_finalize_session_memory",
        lambda **kwargs: captured.update(kwargs) or {"created": True},
    )

    result = phase1_repository.finalize_text_session_memory(
        user_id="user-1", session_id="chat-1", summary_version=1
    )

    assert result == {"created": True}
    assert captured == {
        "user_id": "user-1",
        "session_id": "chat-1",
        "summary_version": 1,
        "session_type": "text",
    }


def test_text_finalize_api_is_owner_scoped(monkeypatch):
    from medrag.app.api import sessions

    captured = {}
    monkeypatch.setattr(
        sessions,
        "finalize_session",
        lambda *args, **kwargs: captured.update(
            {"session_id": args[0], **kwargs}
        )
        or {"created": True},
    )

    result = asyncio.run(
        sessions.finalize_text_session(
            "chat-1",
            summary_version=2,
            current_user=SimpleNamespace(user_id="owner-user"),
        )
    )

    assert result == {"created": True}
    assert captured == {
        "session_id": "chat-1",
        "user_id": "owner-user",
        "summary_version": 2,
    }


def test_text_session_delete_finalizes_before_removal(monkeypatch):
    from medrag.app.api import sessions

    events = []
    monkeypatch.setattr(
        sessions,
        "finalize_session",
        lambda *args, **kwargs: events.append(("finalize", args, kwargs)) or {},
    )
    monkeypatch.setattr(
        sessions,
        "delete_session",
        lambda *args, **kwargs: events.append(("delete", args, kwargs)) or True,
    )

    asyncio.run(
        sessions.remove_session(
            "chat-1",
            current_user=SimpleNamespace(user_id="owner-user", username="alice"),
        )
    )

    assert [event[0] for event in events] == ["finalize", "delete"]


def test_correction_creates_new_confirmed_version_and_supersedes_old(monkeypatch):
    from medrag.infrastructure.storage import phase1_repository

    old = {
        "memory_id": "old-memory",
        "user_id": "user-1",
        "memory_type": "allergy",
        "content": "过敏：青霉素",
        "structured_value": {"value": "青霉素"},
        "status": "confirmed",
        "source_session_type": "voice",
        "source_session_id": "vs_1",
    }
    state = {"fetch": None, "updates": []}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params):
            normalized = " ".join(query.split())
            state["updates"].append((normalized, params))
            if "FOR UPDATE" in normalized:
                state["fetch"] = old
            elif "candidate_key = %s" in normalized:
                state["fetch"] = None
            elif normalized.startswith("INSERT INTO medical_fact_memories"):
                state["fetch"] = {
                    **old,
                    "memory_id": params[0],
                    "content": params[3],
                    "structured_value": {"value": "头孢"},
                    "status": "confirmed",
                    "source_type": "user_correction",
                    "supersedes_memory_id": "old-memory",
                }
            else:
                state["fetch"] = None

        def fetchone(self):
            return state["fetch"]

    class FakeConnection:
        def cursor(self, **kwargs):
            return FakeCursor()

    @contextmanager
    def fake_get_conn():
        yield FakeConnection()

    monkeypatch.setattr(phase1_repository, "get_conn", fake_get_conn)

    replacement = phase1_repository.correct_medical_fact_memory(
        user_id="user-1",
        memory_id="old-memory",
        content="过敏：头孢",
        structured_value={"value": "头孢"},
    )

    assert replacement["status"] == "confirmed"
    assert replacement["supersedes_memory_id"] == "old-memory"
    assert any(
        "SET status = 'superseded'" in query for query, _ in state["updates"]
    )


def test_memory_api_scopes_every_action_to_authenticated_user(monkeypatch):
    from medrag.app.api import memories

    captured = {}
    monkeypatch.setattr(
        memories.phase1_repository,
        "set_medical_fact_status",
        lambda **kwargs: captured.update(kwargs) or {"memory_id": kwargs["memory_id"]},
    )

    result = asyncio.run(
        memories.confirm_memory(
            "memory-1",
            current_user=SimpleNamespace(user_id="owner-user"),
        )
    )

    assert result["memory_id"] == "memory-1"
    assert captured == {
        "user_id": "owner-user",
        "memory_id": "memory-1",
        "status": "confirmed",
    }


def test_reject_correct_delete_and_export_are_owner_scoped(monkeypatch):
    from medrag.app.api import memories

    owner = SimpleNamespace(user_id="owner-user")
    calls = []
    monkeypatch.setattr(
        memories.phase1_repository,
        "set_medical_fact_status",
        lambda **kwargs: calls.append(("reject", kwargs))
        or {"memory_id": kwargs["memory_id"], "status": kwargs["status"]},
    )
    monkeypatch.setattr(
        memories.phase1_repository,
        "correct_medical_fact_memory",
        lambda **kwargs: calls.append(("correct", kwargs))
        or {"memory_id": "replacement", "supersedes_memory_id": kwargs["memory_id"]},
    )
    monkeypatch.setattr(
        memories.phase1_repository,
        "delete_medical_fact_memory",
        lambda **kwargs: calls.append(("delete", kwargs)) or True,
    )
    monkeypatch.setattr(
        memories.phase1_repository,
        "export_controlled_memory",
        lambda **kwargs: calls.append(("export", kwargs))
        or {"schema_version": 1, "medical_fact_memories": []},
    )

    rejected = asyncio.run(memories.reject_memory("memory-1", current_user=owner))
    corrected = asyncio.run(
        memories.correct_memory(
            "memory-1",
            memories.MemoryCorrection(
                content="过敏：头孢",
                structured_value={"value": "头孢"},
            ),
            current_user=owner,
        )
    )
    deleted = asyncio.run(memories.delete_memory("memory-1", current_user=owner))
    exported = asyncio.run(
        memories.export_memories(
            response=SimpleNamespace(headers={}), current_user=owner
        )
    )

    assert rejected["status"] == "rejected"
    assert corrected["supersedes_memory_id"] == "memory-1"
    assert deleted.status_code == 204
    assert exported["schema_version"] == 1
    assert all(call[1]["user_id"] == "owner-user" for call in calls)


def test_voice_memory_context_discloses_only_fresh_confirmed_content(monkeypatch):
    from medrag.app.api import internal_v1

    captured = {}
    monkeypatch.setattr(
        internal_v1,
        "_authorize",
        lambda **kwargs: (None, SimpleNamespace(user_id="owner-user")),
    )
    calls = 0

    def fresh_memories(**kwargs):
        nonlocal calls
        calls += 1
        captured.update(kwargs)
        return [
            {
                "memory_id": "must-not-leak",
                "content": "过敏：青霉素" if calls == 1 else "过敏：头孢",
                "structured_value": {"private": True},
                "source_document_id": "must-not-leak",
                "confidence": 0.99,
            }
        ]

    monkeypatch.setattr(
        internal_v1.phase1_repository,
        "list_medical_fact_memories",
        fresh_memories,
    )
    payload = VoiceMemoryContextRequest(
        session_id="vs_1",
        turn_id="turn_7",
        idempotency_key="phase3-memory-read-7",
    )

    response = asyncio.run(
        internal_v1.get_voice_memory_context(
            "vs_1",
            payload,
            authorization="Bearer short-lived",
            x_request_nonce="00000000-0000-0000-0000-000000000007",
        )
    )
    body = json.loads(response.body)
    second = asyncio.run(
        internal_v1.get_voice_memory_context(
            "vs_1",
            payload,
            authorization="Bearer short-lived",
            x_request_nonce="00000000-0000-0000-0000-000000000008",
        )
    )
    second_body = json.loads(second.body)

    assert body["data"] == {
        "confirmed_facts": ["过敏：青霉素"],
        "fresh": True,
    }
    assert "must-not-leak" not in response.body.decode()
    assert second_body["data"]["confirmed_facts"] == ["过敏：头孢"]
    assert calls == 2
    assert captured == {"user_id": "owner-user", "statuses": ["confirmed"]}
