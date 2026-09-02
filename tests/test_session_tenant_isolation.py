"""Regression tests for cross-user session read, write, and delete isolation."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from medrag.app import session_store
from medrag.infrastructure.storage import postgres_client


def test_get_session_hides_another_users_session(monkeypatch):
    session_get = MagicMock(return_value=None)
    message_list = MagicMock()
    monkeypatch.setattr(session_store, "_pg_session_get", session_get)
    monkeypatch.setattr(session_store, "_pg_message_list", message_list)

    result = session_store.get_session("shared-id", username="bob")

    assert result is None
    session_get.assert_called_once_with("shared-id", "bob")
    message_list.assert_not_called()


def test_add_message_rejects_session_id_owned_by_another_user(monkeypatch):
    message_add = MagicMock(side_effect=PermissionError("another user"))
    monkeypatch.setattr(
        session_store.phase1_repository, "record_text_message", message_add
    )

    with pytest.raises(PermissionError, match="another user"):
        session_store.add_message(
            "shared-id",
            "human",
            "must not be written",
            username="bob",
            user_id="00000000-0000-0000-0000-000000000002",
            turn_id="turn-1",
        )

    message_add.assert_called_once()


def test_add_message_scopes_existing_session_write_to_owner(monkeypatch):
    message_add = MagicMock()
    monkeypatch.setattr(
        session_store.phase1_repository, "record_text_message", message_add
    )

    session_store.add_message(
        "session-1",
        "human",
        "owner message",
        username="alice",
        user_id="00000000-0000-0000-0000-000000000001",
        turn_id="turn-1",
    )

    message_add.assert_called_once_with(
        user_id="00000000-0000-0000-0000-000000000001",
        username="alice",
        session_id="session-1",
        turn_id="turn-1",
        role="user",
        content="owner message",
        rag_trace=None,
    )


def test_delete_session_scopes_delete_to_current_user(monkeypatch):
    session_delete = MagicMock(return_value=False)
    monkeypatch.setattr(session_store, "_pg_session_delete", session_delete)

    deleted = session_store.delete_session("shared-id", username="bob")

    assert deleted is False
    session_delete.assert_called_once_with("shared-id", "bob")


@contextmanager
def _fake_connection(cursor):
    connection = MagicMock()
    connection.cursor.return_value = cursor
    yield connection


def test_postgres_session_read_binds_username(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    monkeypatch.setattr(
        postgres_client,
        "get_conn",
        lambda: _fake_connection(cursor),
    )

    result = postgres_client.session_get("shared-id", "bob")

    assert result is None
    query, params = cursor.execute.call_args.args
    assert "username = %s" in query
    assert params == ("shared-id", "bob")


def test_postgres_session_delete_binds_username(monkeypatch):
    cursor = MagicMock()
    cursor.rowcount = 0
    monkeypatch.setattr(
        postgres_client,
        "get_conn",
        lambda: _fake_connection(cursor),
    )

    deleted = postgres_client.session_delete("shared-id", "bob")

    assert deleted is False
    query, params = cursor.execute.call_args.args
    assert "username = %s" in query
    assert params == ("shared-id", "bob")


def test_postgres_message_write_binds_username(monkeypatch):
    cursor = MagicMock()
    cursor.rowcount = 0
    monkeypatch.setattr(
        postgres_client,
        "get_conn",
        lambda: _fake_connection(cursor),
    )

    written = postgres_client.message_add("shared-id", "bob", "human", "blocked")

    assert written is False
    query, params = cursor.execute.call_args.args
    assert "username = %s" in query
    assert params[-2:] == ("shared-id", "bob")
