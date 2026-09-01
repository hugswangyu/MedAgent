"""阶段 1 user_id、worker JWT、会话绑定与防重放回归测试。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from medrag.app import auth_manager, worker_auth
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


def test_worker_request_consumes_nonce_once_and_binds_session(monkeypatch):
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
    bind = MagicMock()
    monkeypatch.setattr(worker_auth, "decode_worker_token", lambda _: claims)
    monkeypatch.setattr(
        phase1_repository,
        "consume_worker_nonce",
        consume,
    )
    monkeypatch.setattr(phase1_repository, "bind_voice_session", bind)
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
    bind.assert_called_once_with(
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
        "voice_sessions",
        "voice_turns",
        "evidence",
        "audit_events",
        "worker_request_nonces",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in migration
    assert "PRIMARY KEY(session_id, turn_id)" in migration
