import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from medrag.service import chat_service
from medrag.service.chat_service import MedicalChatService


def _make_service(monkeypatch):
    monkeypatch.setattr(chat_service, "get_llm_client", MagicMock())
    monkeypatch.setattr(chat_service, "get_llm_provider", MagicMock())
    return MedicalChatService(
        hybrid_retriever=MagicMock(),
        reranker=MagicMock(),
        prompt_builder=MagicMock(),
        answer_generator=MagicMock(),
        safety_guard=MagicMock(),
        memory_system=MagicMock(),
    )


def test_each_service_initializes_its_own_tool_registry(monkeypatch):
    first = _make_service(monkeypatch)
    second = _make_service(monkeypatch)
    registries = []

    def create_registry():
        registry = MagicMock()
        registries.append(registry)
        return registry

    monkeypatch.setattr("medrag.tools.get_tool_registry", create_registry)

    assert first._tool_registry is None
    assert second._tool_registry is None
    assert first._tool_registry_lock is not second._tool_registry_lock

    assert first._get_tool_registry() is registries[0]
    assert second._tool_registry is None
    assert second._get_tool_registry() is registries[1]
    assert first._tool_registry is registries[0]
    assert len(registries) == 2


def test_service_initialization_order_is_independent(monkeypatch):
    first = _make_service(monkeypatch)
    second = _make_service(monkeypatch)
    calls = 0

    def create_registry():
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr("medrag.tools.get_tool_registry", create_registry)

    second_registry = second._get_tool_registry()
    first_registry = first._get_tool_registry()

    assert second_registry is second._tool_registry
    assert first_registry is first._tool_registry
    assert first_registry is not second_registry
    assert calls == 2


def test_concurrent_first_access_initializes_registry_once(monkeypatch):
    service = _make_service(monkeypatch)
    call_count = 0
    count_lock = threading.Lock()
    registry = object()

    def create_registry():
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.05)
        return registry

    monkeypatch.setattr("medrag.tools.get_tool_registry", create_registry)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: service._get_tool_registry(), range(16)))

    assert call_count == 1
    assert service._tool_registry is registry
    assert all(result is registry for result in results)
