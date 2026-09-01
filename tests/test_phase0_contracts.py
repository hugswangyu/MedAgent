import asyncio
import os
from functools import wraps
from types import SimpleNamespace

import pytest
from medrag.contracts.phase0 import VoiceTurnRecordRequest
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
async def test_input_safety_reports_semantic_dimensions_and_special_population():
    service = Phase0CapabilityService()
    result = await service.input_check(
        "我怀孕了，目前没有胸痛，只想科普胸痛是什么", "req_context"
    )

    assert result["action"] == "allow"
    assert result["semantic_context"] == {
        "current_self": False,
        "current_other": False,
        "other_person": False,
        "historical": False,
        "negated": True,
        "educational": True,
        "ongoing": False,
        "special_populations": ["pregnant"],
    }


@async_test
async def test_input_safety_distinguishes_current_child_from_current_self():
    service = Phase0CapabilityService()
    result = await service.input_check("我孩子现在抽搐", "req_child")

    assert result["action"] == "emergency"
    assert result["semantic_context"]["current_self"] is False
    assert result["semantic_context"]["current_other"] is True
    assert result["semantic_context"]["special_populations"] == ["child"]


@async_test
async def test_ongoing_first_person_red_symptom_is_current_emergency():
    service = Phase0CapabilityService()
    result = await service.input_check(
        "我胸痛一直没有缓解", "req_ongoing"
    )

    assert result["action"] == "emergency"
    assert result["semantic_context"]["ongoing"] is True


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
async def test_output_check_displays_unresolved_conflicts_without_source_priority():
    service = Phase0CapabilityService()
    evidence = [
        {
            "evidence_id": "medical_1",
            "turn_id": "turn_1",
            "source_type": "medical",
            "fact_type": "dosage",
            "fact_subject_id": "drug:aspirin",
            "subject_scope": "general",
            "source_category": "guideline",
            "content_preview": "每日 5 mg",
            "authority_level": "guideline",
        },
        {
            "evidence_id": "personal_1",
            "turn_id": "turn_1",
            "source_type": "personal",
            "fact_type": "dosage",
            "fact_subject_id": "drug:aspirin",
            "subject_scope": "user_specific",
            "source_category": "prescription",
            "content_preview": "每日 10 mg",
            "authority_level": "clinical_document",
        },
    ]

    result = await service.output_check("请按现有处方核实。", evidence, "req_conflict")

    assert result["allowed"] is False
    assert result["conflict_notice"]
    assert result["safe_text"] == result["conflict_notice"]
    conflict = result["evidence_conflicts"][0]
    assert conflict["resolution"] == "unresolved"
    assert conflict["fact_subject_id"] == "drug:aspirin"
    assert conflict["high_risk"] is True
    assert {item["source_type"] for item in conflict["sources"]} == {
        "medical",
        "personal",
    }


@async_test
async def test_output_check_adds_required_dosage_notice_before_tts():
    service = Phase0CapabilityService()
    result = await service.output_check(
        "每日服用 5 mg。", [], "req_dosage"
    )

    assert result["allowed"] is True
    assert "医生处方" in result["safe_text"]
    assert result["required_notices"]


@async_test
async def test_output_check_does_not_conflict_different_dosage_subjects():
    service = Phase0CapabilityService()
    evidence = [
        {
            "evidence_id": "aspirin",
            "turn_id": "turn_1",
            "source_type": "medical",
            "fact_type": "dosage",
            "source_category": "reference",
            "content_preview": "阿司匹林 100mg",
        },
        {
            "evidence_id": "metformin",
            "turn_id": "turn_1",
            "source_type": "medical",
            "fact_type": "dosage",
            "source_category": "reference",
            "content_preview": "二甲双胍 500mg",
        },
    ]

    result = await service.output_check("请分别核对剂量。", evidence, "req")

    assert result["allowed"] is True
    assert result["evidence_conflicts"] == []


def test_turn_record_rejects_cross_turn_evidence():
    with pytest.raises(ValueError, match="request turn_id"):
        VoiceTurnRecordRequest(
            session_id="vs_1",
            turn_id="turn_1",
            idempotency_key="turn-write-123",
            evidence=[
                {
                    "evidence_id": "ev_1",
                    "turn_id": "turn_2",
                    "source_type": "medical",
                    "source_category": "guideline",
                    "source_id": "source_1",
                    "request_id": "req_1",
                    "latency_ms": 1,
                }
            ],
        )


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
