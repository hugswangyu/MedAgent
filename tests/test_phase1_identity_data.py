"""阶段 1 user_id、worker JWT、会话绑定与防重放回归测试。"""

from __future__ import annotations

import uuid
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from medrag.app import auth_manager, worker_auth
from medrag.app.api import control_v1
from medrag.infrastructure.storage import phase1_repository


def test_worker_token_is_short_lived_and_bound_to_user_session_kb(monkeypatch):
    monkeypatch.setenv("MEDRAG_ENV", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "phase1-test-secret")
    user_id = str(uuid.uuid4())
    token = auth_manager.create_worker_token(
        user_id=user_id,
        session_id="vs_1",
        knowledge_base_id="kb_1",
    )

    payload = auth_manager.decode_worker_token(token)

    assert payload["sub"] == user_id
    assert payload["sid"] == "vs_1"
    assert payload["kid"] == "kb_1"
    assert payload["aud"] == auth_manager.WORKER_AUDIENCE
    assert payload["exp"] - payload["iat"] <= 300


def test_worker_request_requires_preclaimed_binding_and_consumes_nonce_once(monkeypatch):
    user_id = str(uuid.uuid4())
    token_jti = str(uuid.uuid4())
    nonce = str(uuid.uuid4())
    claims = {
        "sub": user_id,
        "sid": "vs_1",
        "kid": "kb_1",
        "scope": "safety:input",
        "jti": token_jti,
        "exp": datetime.now(timezone.utc).timestamp() + 300,
    }
    consume = MagicMock(side_effect=[True, False])
    validate = MagicMock(return_value=True)
    monkeypatch.setattr(worker_auth, "decode_worker_token", lambda _: claims)
    monkeypatch.setattr(
        phase1_repository,
        "consume_worker_nonce",
        consume,
    )
    monkeypatch.setattr(
        phase1_repository, "validate_claimed_worker_binding", validate
    )
    payload = {
        "session_id": "vs_1",
        "turn_id": "turn_1",
        "idempotency_key": "idem-key",
        "text": "测试",
    }

    principal = worker_auth.authorize_worker_request(
        authorization="Bearer signed",
        nonce=nonce,
        required_scope="safety:input",
        payload=payload,
    )

    assert principal.user_id == user_id
    validate.assert_called_with(
        user_id=user_id,
        session_id="vs_1",
        knowledge_base_id="kb_1",
    )
    with pytest.raises(worker_auth.WorkerAuthorizationError) as exc_info:
        worker_auth.authorize_worker_request(
            authorization="Bearer signed",
            nonce=nonce,
            required_scope="safety:input",
            payload=payload,
        )
    assert exc_info.value.code == "REPLAY_DETECTED"


def test_worker_request_rejects_unclaimed_binding_before_nonce(monkeypatch):
    claims = {
        "sub": str(uuid.uuid4()),
        "sid": "vs_1",
        "kid": "kb_1",
        "scope": "safety:input",
        "jti": str(uuid.uuid4()),
        "exp": datetime.now(timezone.utc).timestamp() + 300,
    }
    monkeypatch.setattr(worker_auth, "decode_worker_token", lambda _: claims)
    monkeypatch.setattr(
        phase1_repository, "validate_claimed_worker_binding", lambda **_: False
    )
    consume = MagicMock()
    monkeypatch.setattr(phase1_repository, "consume_worker_nonce", consume)

    with pytest.raises(worker_auth.WorkerAuthorizationError) as exc_info:
        worker_auth.authorize_worker_request(
            authorization="Bearer signed",
            nonce=str(uuid.uuid4()),
            required_scope="safety:input",
            payload={"session_id": "vs_1"},
        )

    assert exc_info.value.code == "SESSION_BINDING_MISMATCH"
    consume.assert_not_called()


def test_worker_request_rejects_session_binding_mismatch(monkeypatch):
    monkeypatch.setattr(
        worker_auth,
        "decode_worker_token",
        lambda _: {
            "sub": str(uuid.uuid4()),
            "sid": "vs_owner",
            "kid": "kb_1",
            "scope": "medical:retrieve",
            "jti": str(uuid.uuid4()),
            "exp": datetime.now(timezone.utc).timestamp() + 300,
        },
    )

    with pytest.raises(worker_auth.WorkerAuthorizationError) as exc_info:
        worker_auth.authorize_worker_request(
            authorization="Bearer signed",
            nonce=str(uuid.uuid4()),
            required_scope="medical:retrieve",
            payload={"session_id": "vs_other"},
        )

    assert exc_info.value.code == "SESSION_BINDING_MISMATCH"


def test_phase1_migration_declares_required_postgres_tables():
    migration = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "phase1_identity_data.sql"
    ).read_text(encoding="utf-8")

    for table in (
        "users",
        "knowledge_base_ownership",
        "voice_sessions",
        "voice_turns",
        "evidence",
        "audit_events",
        "worker_request_nonces",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in migration
    assert "PRIMARY KEY(session_id, turn_id)" in migration
    assert "binding_version" in migration
    assert "livekit_job_id" in migration
    assert "idx_voice_sessions_one_open_per_user" in migration


@pytest.mark.parametrize(
    ("left", "right"),
    [("Alice", "alice"), ("Ａlice", "Alice"), ("e\u0301", "é")],
)
def test_username_normalization_detects_case_and_unicode_equivalence(left, right):
    assert phase1_repository.normalize_username(left) == phase1_repository.normalize_username(right)


def test_worker_bootstrap_token_contains_only_binding_coordinates(monkeypatch):
    monkeypatch.setenv("MEDRAG_ENV", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "phase1-test-secret")
    token = auth_manager.create_worker_bootstrap_token(
        session_id="vs_1", binding_version=7
    )

    claims = auth_manager.decode_worker_bootstrap_token(token)

    assert claims["sid"] == "vs_1"
    assert claims["ver"] == 7
    assert "sub" not in claims
    assert "kid" not in claims


def test_worker_claim_uses_postgres_cas_and_server_binding(monkeypatch):
    user_id = str(uuid.uuid4())
    monkeypatch.setattr(
        control_v1,
        "decode_worker_bootstrap_token",
        lambda _: {"sid": "vs_1", "ver": 4},
    )
    claim = MagicMock(
        return_value={
            "session_id": "vs_1",
            "user_id": user_id,
            "knowledge_base_id": "kb_pg",
        }
    )
    monkeypatch.setattr(phase1_repository, "claim_voice_session_binding", claim)
    monkeypatch.setattr(control_v1, "create_worker_token", lambda **_: "worker-token")

    result = asyncio.run(
        control_v1.claim_voice_session(
            control_v1.VoiceSessionClaim(
                session_id="vs_1",
                binding_version=4,
                room_name="room_1",
                livekit_job_id="job_1",
            ),
            type("Credentials", (), {"credentials": "bootstrap"})(),
        )
    )

    claim.assert_called_once_with(
        session_id="vs_1",
        expected_version=4,
        room_name="room_1",
        livekit_job_id="job_1",
    )
    assert result["worker_token"] == "worker-token"
    assert result["knowledge_base_id"] == "kb_pg"


def test_control_plane_routes_are_not_anonymous():
    from fastapi.testclient import TestClient
    from medrag.app.server import app

    client = TestClient(app)
    assert client.get("/control/v1/knowledge-bases").status_code in {401, 403}
    assert client.post(
        "/control/v1/worker/voice-sessions/claim",
        json={
            "session_id": "vs_1",
            "binding_version": 1,
            "room_name": "room_1",
            "livekit_job_id": "job_1",
        },
    ).status_code == 401


def test_control_plane_mutations_require_separate_service_key(monkeypatch):
    monkeypatch.setenv("MEDAGENT_CONTROL_PLANE_KEY", "service-secret")

    control_v1.require_control_plane_key("service-secret")
    with pytest.raises(HTTPException) as exc_info:
        control_v1.require_control_plane_key("user-supplied-wrong-key")

    assert exc_info.value.status_code == 401
