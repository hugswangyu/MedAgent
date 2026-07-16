import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from medrag.app import document_store
from medrag.data import user_case_store
from medrag.infrastructure.storage import JsonStore


class _ReadCollisionStore(JsonStore):
    """Force legacy separate read/write callers to read the same snapshot."""

    def __init__(self, path):
        super().__init__(path)
        self._read_barrier = threading.Barrier(2)

    def read(self):
        data = super().read()
        self._read_barrier.wait(timeout=5)
        return data


def _run_concurrently(*operations):
    with ThreadPoolExecutor(max_workers=len(operations)) as executor:
        futures = [executor.submit(operation) for operation in operations]
        return [future.result(timeout=10) for future in futures]


def test_concurrent_document_adds_are_not_lost(tmp_path, monkeypatch):
    path = tmp_path / "documents.json"
    monkeypatch.setattr(document_store, "_doc_store", _ReadCollisionStore(str(path)))

    _run_concurrently(
        lambda: document_store.add_document("first.pdf", username="alice"),
        lambda: document_store.add_document("second.pdf", username="alice"),
    )

    assert {item["filename"] for item in JsonStore(str(path)).read()} == {
        "first.pdf",
        "second.pdf",
    }


def test_concurrent_user_case_adds_are_not_lost(tmp_path, monkeypatch):
    path = tmp_path / "cases.json"
    monkeypatch.setattr(user_case_store, "_case_store", _ReadCollisionStore(str(path)))

    _run_concurrently(
        lambda: user_case_store.add_user_case("alice", "first.pdf", ["one"]),
        lambda: user_case_store.add_user_case("alice", "second.pdf", ["two"]),
    )

    assert {item["filename"] for item in JsonStore(str(path)).read()} == {
        "first.pdf",
        "second.pdf",
    }


def test_remove_does_not_overwrite_concurrent_add(tmp_path, monkeypatch):
    path = tmp_path / "documents.json"
    store = _ReadCollisionStore(str(path))
    JsonStore(str(path)).write([{"filename": "old.pdf", "username": "alice"}])
    monkeypatch.setattr(document_store, "_doc_store", store)

    results = _run_concurrently(
        lambda: document_store.remove_document("old.pdf", username="alice"),
        lambda: document_store.add_document("new.pdf", username="alice"),
    )

    assert results[0] is True
    assert [item["filename"] for item in JsonStore(str(path)).read()] == ["new.pdf"]


def test_normalized_path_uses_one_process_lock(tmp_path):
    path = tmp_path / "index.json"
    first = JsonStore(str(path))
    second = JsonStore(os.path.join(str(path.parent), "unused", "..", path.name))

    assert os.path.isabs(first.path)
    assert first.path == second.path
    assert first._lock is second._lock


def test_mutator_failure_preserves_original_file(tmp_path):
    path = tmp_path / "index.json"
    original = b'[{"filename": "original.pdf"}]\n'
    path.write_bytes(original)
    store = JsonStore(str(path))

    def fail(_items):
        raise RuntimeError("mutation failed")

    with pytest.raises(RuntimeError, match="mutation failed"):
        store.update(fail, default_factory=list)

    assert path.read_bytes() == original


def test_serialization_failure_preserves_original_file(tmp_path):
    path = tmp_path / "index.json"
    original = b'[{"filename": "original.pdf"}]\n'
    path.write_bytes(original)
    store = JsonStore(str(path))

    def add_unserializable(items):
        items.append(object())

    with pytest.raises(TypeError):
        store.update(add_unserializable, default_factory=list)

    assert path.read_bytes() == original


def test_replace_failure_preserves_original_file(tmp_path, monkeypatch):
    path = tmp_path / "index.json"
    original = b'[{"filename": "original.pdf"}]\n'
    path.write_bytes(original)
    store = JsonStore(str(path))

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.update(lambda items: items.append({"filename": "new.pdf"}), default_factory=list)

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_json_aborts_update_without_overwrite(tmp_path):
    path = tmp_path / "index.json"
    original = b"{broken"
    path.write_bytes(original)
    store = JsonStore(str(path))

    with pytest.raises(json.JSONDecodeError):
        store.update(lambda data: data, default_factory=list)

    assert path.read_bytes() == original
