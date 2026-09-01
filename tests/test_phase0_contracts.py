import asyncio
import os
from functools import wraps
from types import SimpleNamespace

from medrag.service.phase0_capabilities import Phase0CapabilityService


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


@async_test
async def test_input_safety_distinguishes_contexts():
    service = Phase0CapabilityService()
    education = await service.input_check("胸痛是什么", "req_1")
    emergency = await service.input_check(
        "我现在胸痛而且呼吸困难", "req_2"
    )
    negated = await service.input_check("我没有胸痛", "req_3")
    mixed = await service.input_check(
        "我没有胸痛，但我现在呼吸困难", "req_4"
    )

    assert education["action"] == "allow"
    assert emergency["action"] == "emergency"
    assert "120" in emergency["fixed_response"]
    assert negated["action"] == "allow"
    assert mixed["action"] == "emergency"


@async_test
async def test_input_safety_prioritizes_current_self_in_mixed_context():
    service = Phase0CapabilityService()

    mixed = await service.input_check(
        "我朋友以前胸痛，但我现在呼吸困难", "req_mixed"
    )
    other_only = await service.input_check(
        "我现在想问，我朋友以前胸痛", "req_other"
    )

    assert mixed["action"] == "emergency"
    assert "呼吸困难" in mixed["risk_types"]
    assert "120" in mixed["fixed_response"]
    assert other_only["action"] == "allow"
    assert other_only["risk_level"] == "context_only"


@async_test
async def test_retrieve_is_retrieval_only_and_excludes_personal_case():
    class Router:
        def route(self, query, use_llm=True):
            assert use_llm is False
            return {
                "use_kg": True,
                "use_qa": True,
                "needs_case_context": True,
            }

    class Retriever:
        router = Router()

        def retrieve(
            self, query, top_k, department, username, route
        ):
            assert username is None
            assert route["needs_case_context"] is False
            return {
                "route": route,
                "kg_results": [
                    {"id": "kg-1", "description": "医学事实"}
                ],
                "qa_results": [{"id": "qa-1", "answer": "医学问答"}],
                "case_results": [{"id": "must-not-leak"}],
            }

    service = Phase0CapabilityService(
        SimpleNamespace(hybrid_retriever=Retriever())
    )
    result = await service.retrieve_medical(
        query="高血压",
        top_k=5,
        department=None,
        turn_id="turn-1",
        request_id="req-1",
    )
    assert result["retrieval_only"] is True
    assert result["evidence_count"] == 2
    assert {
        item["source_type"] for item in result["evidence"]
    } == {"medical"}


@async_test
async def test_idempotency_replay_and_conflict():
    service = Phase0CapabilityService()
    calls = 0

    async def operation(request_id):
        nonlocal calls
        calls += 1
        return {"request_id_seen": request_id}

    first = await service.invoke(
        operation="test",
        idempotency_key="same-key-123",
        payload={"x": 1},
        timeout_ms=100,
        function=operation,
    )
    replay = await service.invoke(
        operation="test",
        idempotency_key="same-key-123",
        payload={"x": 1},
        timeout_ms=100,
        function=operation,
    )
    conflict = await service.invoke(
        operation="test",
        idempotency_key="same-key-123",
        payload={"x": 2},
        timeout_ms=100,
        function=operation,
    )
    assert first.request_id == replay.request_id
    assert replay.metrics["idempotency_replay"] is True
    assert calls == 1
    assert conflict.error.code == "IDEMPOTENCY_CONFLICT"


@async_test
async def test_output_check_fails_closed_for_dangerous_advice():
    service = Phase0CapabilityService()
    result = await service.output_check(
        "不用去急诊，在家等一等。", [], "req-1"
    )
    assert result["allowed"] is False
    assert "未通过" in result["safe_text"]


@async_test
async def test_deterministic_tool_contract():
    service = Phase0CapabilityService()
    result = await service.execute_tool(
        "guide_department", {"query": "胸痛挂什么科"}, "req-1"
    )
    assert result["tool_name"] == "guide_department"
    assert result["deterministic"] is True
    assert "急诊" in result["result"]


def test_internal_api_auth_and_validation_use_frozen_envelope():
    from fastapi.testclient import TestClient

    from medrag.app.server import app

    os.environ["MEDAGENT_INTERNAL_API_KEY"] = "test-internal-secret"
    client = TestClient(app)
    unauthorized = client.post(
        "/internal/v1/safety/input-check",
        headers={"X-Internal-API-Key": "wrong"},
        json={
            "session_id": "session_1",
            "turn_id": "turn_1",
            "idempotency_key": "input-key-123",
            "text": "胸痛是什么",
        },
    )
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    invalid = client.post(
        "/internal/v1/safety/input-check",
        headers={"X-Internal-API-Key": "test-internal-secret"},
        json={"text": "胸痛是什么"},
    )
    assert invalid.status_code == 422
    body = invalid.json()
    assert set(body) == {
        "request_id",
        "status",
        "data",
        "metrics",
        "error",
    }
    assert body["error"]["code"] == "INVALID_REQUEST"


@async_test
async def test_concurrent_idempotent_requests_execute_once():
    service = Phase0CapabilityService()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def operation(request_id):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"request_id_seen": request_id}

    first = asyncio.create_task(
        service.invoke(
            operation="single-flight",
            idempotency_key="concurrent-key-123",
            payload={"x": 1},
            timeout_ms=1000,
            function=operation,
        )
    )
    await started.wait()
    second = asyncio.create_task(
        service.invoke(
            operation="single-flight",
            idempotency_key="concurrent-key-123",
            payload={"x": 1},
            timeout_ms=1000,
            function=operation,
        )
    )
    await asyncio.sleep(0)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert calls == 1
    assert first_result.request_id == second_result.request_id
    assert second_result.metrics["idempotency_waited"] is True
