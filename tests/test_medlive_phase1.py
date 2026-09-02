"""阶段 1 知识库所有权、Voice Session 绑定和 worker token 测试。"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from medlive.agent.tool.medical_client import MedicalCapabilityClient
from medlive.main import (
    _close_binding_then_compact_history,
    _start_worker_lifecycle,
    _voice_session_from_metadata,
)
from medlive.security import ALGORITHM, CurrentUser, decode_access_token, get_current_user
from medlive.voice.lifecycle import WorkerSessionLifecycle


def test_access_jwt_sub_is_user_id(monkeypatch):
    secret = "phase1-shared-test-secret"
    monkeypatch.setenv("JWT_SECRET_KEY", secret)
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": user_id,
            "username": "alice",
            "token_use": "access",
            "is_admin": True,
            "ver": 3,
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        secret,
        algorithm=ALGORITHM,
    )

    user = decode_access_token(token)

    assert user.user_id == user_id
    assert user.username == "alice"
    assert user.is_admin is True
    assert user.token_version == 3


async def test_voice_session_metadata_claims_server_binding():
    user_id = str(uuid.uuid4())
    captured = {}

    class FakeControlPlane:
        async def claim_voice_session(self, token, payload):
            captured.update({"token": token, "payload": payload})
            return {
                "session_id": "vs_1",
                "user_id": user_id,
                "knowledge_base_id": "kb_owner",
                "worker_token": "server-worker-token",
            }

    ctx = SimpleNamespace(
        room=SimpleNamespace(name="room_owner", metadata=""),
        job=SimpleNamespace(
            id="job_1",
            metadata='{"session_id":"vs_1","binding_version":1,'
            '"worker_bootstrap_token":"bootstrap"}',
        ),
    )

    bound = await _voice_session_from_metadata(ctx, FakeControlPlane())

    assert bound["user_id"] == user_id
    assert bound["knowledge_base_id"] == "kb_owner"
    assert captured == {
        "token": "bootstrap",
        "payload": {
            "session_id": "vs_1",
            "binding_version": 1,
            "room_name": "room_owner",
            "livekit_job_id": "job_1",
        },
    }


@pytest.mark.parametrize("metadata", ["", "not-json", '{"binding_version":1}'])
async def test_worker_metadata_is_fail_closed_by_default(monkeypatch, metadata):
    monkeypatch.delenv("LIVERAG_ALLOW_UNBOUND_WORKER", raising=False)
    monkeypatch.setenv("LIVERAG_ENV", "prod")
    ctx = SimpleNamespace(
        room=SimpleNamespace(name="room_1", metadata=""),
        job=SimpleNamespace(id="job_1", metadata=metadata),
    )

    with pytest.raises(RuntimeError):
        await _voice_session_from_metadata(ctx, SimpleNamespace())


async def test_unbound_worker_requires_explicit_development_switch(monkeypatch):
    monkeypatch.setenv("LIVERAG_ENV", "dev")
    monkeypatch.setenv("LIVERAG_ALLOW_UNBOUND_WORKER", "true")
    ctx = SimpleNamespace(
        room=SimpleNamespace(name="room_1", metadata=""),
        job=SimpleNamespace(id="job_1", metadata=""),
    )

    assert await _voice_session_from_metadata(ctx, SimpleNamespace()) is None


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/model/config"),
        ("get", "/model/context-config"),
        ("get", "/prompt/soul"),
        ("get", "/session/messages"),
        ("get", "/session/rag-context"),
        ("get", "/session/turns"),
        ("post", "/session/clear"),
        ("get", "/rag/config"),
    ],
)
def test_global_data_and_config_endpoints_require_authentication(method, path):
    from medlive.api.server import app

    response = getattr(TestClient(app), method)(path)

    assert response.status_code in {401, 403}


def test_non_admin_cannot_write_global_config():
    from medlive.api.server import app

    async def ordinary_user():
        return CurrentUser(
            user_id=str(uuid.uuid4()),
            username="alice",
            access_token="access",
        )

    app.dependency_overrides[get_current_user] = ordinary_user
    try:
        response = TestClient(app).put("/model/config", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403


async def test_knowledge_base_list_uses_control_plane_ownership(monkeypatch):
    from medlive.api import server

    class FakeControlPlane:
        async def list_knowledge_bases(self, access_token):
            assert access_token == "access"
            return {"knowledge_bases": [{"kb_id": "kb_pg"}], "total": 1}

    monkeypatch.setattr(server, "control_plane", FakeControlPlane())
    monkeypatch.setattr(
        server.metadata_store,
        "public_knowledge_base_detail",
        lambda kb_id: {"kb_id": kb_id, "name": "PG-owned"},
    )

    response = await server.rag_knowledge_bases(
        CurrentUser(
            user_id=str(uuid.uuid4()),
            username="alice",
            access_token="access",
        )
    )

    assert b'"kb_id":"kb_pg"' in response.body


def test_worker_entrypoint_has_no_sqlite_session_store_or_self_signed_token():
    from pathlib import Path

    import medlive.main

    source = Path(medlive.main.__file__).read_text(encoding="utf-8")
    assert "VoiceSessionStore" not in source
    assert "create_worker_token" not in source


async def test_claim_is_released_if_lifecycle_registration_fails():
    calls = []

    class FakeControlPlane:
        async def refresh_worker_token(self, token):
            calls.append(("refresh", token))
            return "fresh-token"

        async def end_worker_session(self, token):
            calls.append(("end", token))

    class BrokenContext:
        def add_shutdown_callback(self, callback):
            raise RuntimeError("callback registration failed")

    with pytest.raises(RuntimeError, match="callback registration failed"):
        await _start_worker_lifecycle(
            BrokenContext(),
            FakeControlPlane(),
            {"worker_token": "claimed-token"},
        )

    assert calls == [("refresh", "claimed-token"), ("end", "fresh-token")]


async def test_early_shutdown_callback_flushes_audit_before_binding_release():
    events = []
    callbacks = []

    class FakeControlPlane:
        async def refresh_worker_token(self, token):
            events.append(("refresh", token))
            return "fresh-token"

        async def end_worker_session(self, token):
            events.append(("end", token))

    class FakeContext:
        def add_shutdown_callback(self, callback):
            callbacks.append(callback)

    coordinator = await _start_worker_lifecycle(
        FakeContext(),
        FakeControlPlane(),
        {"worker_token": "claimed-token"},
    )
    assert coordinator is not None

    async def flush_audit():
        events.append(("flush", None))

    coordinator.attach_audit_flush(flush_audit)

    await callbacks[0]()
    await coordinator.close()

    assert events == [
        ("flush", None),
        ("refresh", "claimed-token"),
        ("end", "fresh-token"),
    ]


async def test_background_heartbeat_refreshes_without_capability_calls():
    refreshed = asyncio.Event()
    calls = []

    class FakeControlPlane:
        async def refresh_worker_token(self, token):
            calls.append(("refresh", token))
            refreshed.set()
            return "renewed-token"

        async def end_worker_session(self, token):
            calls.append(("end", token))

    lifecycle = WorkerSessionLifecycle(
        FakeControlPlane(), "initial-token", heartbeat_interval_s=0.01
    )
    lifecycle.start()
    await asyncio.wait_for(refreshed.wait(), timeout=1)

    assert await lifecycle.current_token() == "renewed-token"
    await lifecycle.close()
    assert calls[-1] == ("end", "renewed-token")


async def test_bound_session_never_falls_back_to_independent_history():
    events = []

    class FakeLifecycle:
        async def close(self):
            events.append("end")

    class CompactorMustNotRun:
        async def compact_after_call(self, **kwargs):
            events.append("compact")
            raise AssertionError("PostgreSQL is the sole fact source")

    result = await _close_binding_then_compact_history(
        FakeLifecycle(),
        CompactorMustNotRun(),
        {"kb_id": "kb_1", "name": "KB"},
    )

    assert events == ["end"]
    assert result["reason"] == "postgres_memory_replacement_unavailable"


async def test_verified_postgres_memory_replacement_disables_independent_history():
    events = []

    class FakeLifecycle:
        async def close(self):
            events.append("end")

    class CompactorMustNotRun:
        async def compact_after_call(self, **kwargs):
            events.append("compact")
            raise AssertionError("legacy history must be disabled")

    result = await _close_binding_then_compact_history(
        FakeLifecycle(),
        CompactorMustNotRun(),
        {"kb_id": "kb_1", "name": "KB"},
        replacement_result={
            "replacement_verified": True,
            "summary_id": "summary_1",
            "summary_version": 1,
        },
    )

    assert events == ["end"]
    assert result == {
        "updated": False,
        "reason": "postgres_memory_replacement_verified",
        "summary_id": "summary_1",
        "summary_version": 1,
    }


async def test_unbound_development_session_also_cannot_use_independent_history():
    class CompactorMustNotRun:
        async def compact_after_call(self, **kwargs):
            raise AssertionError("PostgreSQL remains the sole fact source")

    result = await _close_binding_then_compact_history(
        None,
        CompactorMustNotRun(),
        {"kb_id": "kb_1", "name": "KB"},
    )

    assert result["reason"] == "medlive_independent_history_disabled"


async def test_voice_session_service_end_path_cannot_write_independent_history():
    from medlive.voice.session import VoiceSessionService

    events = []

    class FakeControlPlane:
        async def get_voice_session(self, access_token, session_id):
            assert access_token == "access-token"
            return {
                "session_id": session_id,
                "status": "active",
                "knowledge_base_id": "kb_1",
            }

        async def end_voice_session(self, access_token, session_id):
            events.append((access_token, session_id))

    class Service:
        control_plane = FakeControlPlane()

        @staticmethod
        def _attach_kb_name(session):
            session["kb_name"] = "KB"

        @staticmethod
        def public_session_detail(session):
            return dict(session)

    result = await VoiceSessionService.end_session(
        Service(),
        "vs_1",
        access_token="access-token",
    )

    assert events == [("access-token", "vs_1")]
    assert result["status"] == "ended"
    assert result["history_compaction"] == {
        "updated": False,
        "reason": "medlive_independent_history_disabled",
    }


def test_bound_session_prompt_does_not_read_legacy_history():
    from medlive.context.renderer import SessionPromptRenderer

    class Store:
        def read_history(self, *args, **kwargs):
            raise AssertionError("bound Phase 3 session must not read history.jsonl")

        def read_system_prompt_template(self):
            return "{{SOUL_MD}}\n{{HISTORY_JSONL}}\n{{KNOWLEDGE_OVERVIEW_MD}}\n{{RAG_TOOL_DESCRIPTION}}\n{{KB_ID}}\n{{KB_NAME}}"

        def read_soul(self):
            return ""

        def read_knowledge_overview(self, kb_id):
            return ""

        def write_session_system_prompt(self, prompt):
            assert "过敏：" not in prompt

    result = SessionPromptRenderer(store=Store(), history_limit=8).render(
        kb_id="kb_1",
        kb_name="KB",
        rag_tool_mode="auto",
    )

    assert result.history_count == 0
    assert "独立 history 已停用" in result.prompt


async def test_medical_client_uses_worker_bearer_and_unique_nonce():
    from aiohttp import web

    captured: list[dict[str, str | None]] = []

    async def handler(request):
        captured.append(
            {
                "authorization": request.headers.get("Authorization"),
                "nonce": request.headers.get("X-Request-Nonce"),
                "api_key": request.headers.get("X-Internal-API-Key"),
            }
        )
        return web.json_response(
            {
                "request_id": f"req_{len(captured)}",
                "status": "ok",
                "data": {"action": "allow"},
                "metrics": {},
                "error": None,
            }
        )

    app = web.Application()
    app.router.add_post("/internal/v1/safety/input-check", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = MedicalCapabilityClient(
        base_url=f"http://127.0.0.1:{port}",
        worker_token_factory=lambda: "short-lived-token",
        session_id="vs_1",
    )
    try:
        await client.input_check(text="测试", turn_id="turn_1")
        await client.input_check(text="测试", turn_id="turn_1")
    finally:
        await client.aclose()
        await runner.cleanup()

    assert all(item["authorization"] == "Bearer short-lived-token" for item in captured)
    assert all(item["api_key"] is None for item in captured)
    assert uuid.UUID(str(captured[0]["nonce"]))
    assert captured[0]["nonce"] != captured[1]["nonce"]


def test_ordinary_user_can_read_owned_voice_session_evidence(monkeypatch):
    from medlive.api import server

    user_id = str(uuid.uuid4())
    calls = []

    class FakeVoiceSessionService:
        async def paginate_turns(
            self, session_id, *, user_id, limit, cursor, access_token
        ):
            calls.append(("turns", session_id, user_id, access_token, limit, cursor))
            return {
                "items": [{"turn_index": 1, "rag": {"evidence_count": 1}}],
                "has_more": False,
            }

        async def paginate_rag_context(
            self, session_id, *, user_id, limit, cursor, access_token
        ):
            calls.append(
                ("rag-context", session_id, user_id, access_token, limit, cursor)
            )
            return {
                "items": [{"turn_index": 1, "evidence_chunks": [{"chunk_id": "c1"}]}],
                "has_more": False,
            }

    async def ordinary_user():
        return CurrentUser(
            user_id=user_id,
            username="alice",
            access_token="ordinary-user-token",
        )

    monkeypatch.setattr(
        server, "_voice_session_service", lambda: FakeVoiceSessionService()
    )
    server.app.dependency_overrides[get_current_user] = ordinary_user
    try:
        client = TestClient(server.app)
        turns = client.get("/voice/sessions/vs_owned/turns?limit=20")
        rag_context = client.get(
            "/voice/sessions/vs_owned/rag-context?limit=20"
        )
    finally:
        server.app.dependency_overrides.clear()

    assert turns.status_code == 200
    assert turns.json()["data"]["items"][0]["turn_index"] == 1
    assert rag_context.status_code == 200
    assert rag_context.json()["data"]["items"][0]["evidence_chunks"][0]["chunk_id"] == "c1"
    assert calls == [
        ("turns", "vs_owned", user_id, "ordinary-user-token", 20, None),
        ("rag-context", "vs_owned", user_id, "ordinary-user-token", 20, None),
    ]


def test_ordinary_user_cannot_read_another_users_voice_session(monkeypatch):
    from medlive.api import server

    class FakeVoiceSessionService:
        async def paginate_turns(self, session_id, **kwargs):
            raise KeyError(session_id)

        async def paginate_rag_context(self, session_id, **kwargs):
            raise KeyError(session_id)

    async def ordinary_user():
        return CurrentUser(
            user_id=str(uuid.uuid4()),
            username="alice",
            access_token="ordinary-user-token",
        )

    monkeypatch.setattr(
        server, "_voice_session_service", lambda: FakeVoiceSessionService()
    )
    server.app.dependency_overrides[get_current_user] = ordinary_user
    try:
        client = TestClient(server.app)
        turns = client.get("/voice/sessions/vs_other/turns")
        rag_context = client.get("/voice/sessions/vs_other/rag-context")
    finally:
        server.app.dependency_overrides.clear()

    assert turns.status_code == 404
    assert rag_context.status_code == 404
    assert "vs_other" not in str(turns.json().get("data"))
    assert "vs_other" not in str(rag_context.json().get("data"))
