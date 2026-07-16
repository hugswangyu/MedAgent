"""Authentication configuration and JWT security regression tests."""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from medrag.app import auth_manager


def test_prod_rejects_missing_jwt_secret(monkeypatch):
    monkeypatch.setenv("MEDRAG_ENV", "prod")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY is required"):
        auth_manager.load_auth_config()


@pytest.mark.parametrize(
    "secret",
    [auth_manager.DEFAULT_DEV_SECRET_KEY, "too-short"],
)
def test_prod_rejects_weak_jwt_secret(monkeypatch, secret):
    monkeypatch.setenv("MEDRAG_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", secret)

    with pytest.raises(RuntimeError):
        auth_manager.load_auth_config()


def test_prod_accepts_strong_jwt_secret_without_logging_it(monkeypatch, caplog):
    secret = "a-strong-production-secret-key-value-12345"
    monkeypatch.setenv("MEDRAG_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", secret)

    with caplog.at_level(logging.INFO, logger=auth_manager.__name__):
        config = auth_manager.load_auth_config()

    assert config.environment == "prod"
    assert config.secret_key == secret
    assert "jwt_secret=configured" in caplog.text
    assert secret not in caplog.text


def test_expired_token_is_rejected(monkeypatch):
    monkeypatch.setenv("MEDRAG_ENV", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key")
    token = auth_manager.create_access_token(
        "alice",
        expires_delta=timedelta(seconds=-1),
    )

    assert auth_manager.decode_access_token(token) is None


def test_token_with_wrong_signature_is_rejected(monkeypatch):
    monkeypatch.setenv("MEDRAG_ENV", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "signing-key-one")
    token = auth_manager.create_access_token("alice")
    monkeypatch.setenv("JWT_SECRET_KEY", "different-signing-key")

    assert auth_manager.decode_access_token(token) is None


def test_valid_token_round_trip(monkeypatch):
    monkeypatch.setenv("MEDRAG_ENV", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key")

    token = auth_manager.create_access_token("alice")

    assert auth_manager.decode_access_token(token)["sub"] == "alice"
