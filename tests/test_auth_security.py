"""Authentication configuration and JWT security regression tests."""

from __future__ import annotations

import logging
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from medrag.app import auth_manager
from medrag.app.api import auth as auth_api
from medrag.app.schemas import LoginRequest
from medrag.auth import credentials


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


def test_public_registration_defaults_off_in_prod(monkeypatch):
    monkeypatch.delenv("ALLOW_PUBLIC_REGISTRATION", raising=False)

    config = auth_manager.load_auth_config(
        environment="prod",
        secret_key="a-strong-production-secret-key-value-12345",
    )

    assert config.allow_public_registration is False


def test_public_registration_can_be_explicitly_enabled_in_prod():
    config = auth_manager.load_auth_config(
        environment="prod",
        secret_key="a-strong-production-secret-key-value-12345",
        allow_public_registration="true",
    )

    assert config.allow_public_registration is True


def test_register_rejects_requests_when_public_registration_is_disabled(monkeypatch):
    monkeypatch.setattr(auth_api, "is_public_registration_enabled", lambda: False)
    create_user = MagicMock()
    monkeypatch.setattr(auth_api, "create_user", create_user)

    with pytest.raises(HTTPException) as exc_info:
        auth_api.register(LoginRequest(username="alice", password="secret"))

    assert exc_info.value.status_code == 403
    create_user.assert_not_called()


def test_init_auth_does_not_create_default_admin(monkeypatch):
    monkeypatch.setattr(auth_manager, "load_auth_config", MagicMock())
    monkeypatch.setattr(auth_manager, "load_credentials", MagicMock(return_value={}))
    save_credentials = MagicMock()
    monkeypatch.setattr(auth_manager, "save_credentials", save_credentials)

    auth_manager.init_auth()

    save_credentials.assert_not_called()


def test_legacy_credentials_helper_does_not_create_default_admin(monkeypatch):
    monkeypatch.setattr(credentials, "load_credentials", MagicMock(return_value={}))
    save_credentials = MagicMock()
    monkeypatch.setattr(credentials, "save_credentials", save_credentials)

    assert credentials.get_or_create_credentials() == {}
    save_credentials.assert_not_called()
