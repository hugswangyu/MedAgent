"""阶段 0 内部能力 HTTP API。"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from medrag.app.worker_auth import (
    WorkerAuthorizationError,
    WorkerPrincipal,
    allow_legacy_internal_key,
    authorize_worker_request,
)
from medcontracts.phase0 import (
    CapabilityEnvelope,
    InputCheckRequest,
    MedicalRetrieveRequest,
    MedicalToolRequest,
    OutputCheckRequest,
    VoiceMemoryContextRequest,
    VoiceSessionFinalizeRequest,
    VoiceTurnRecordRequest,
)
from medrag.infrastructure.storage import phase1_repository
from medrag.service.phase0_capabilities import Phase0CapabilityService

router = APIRouter()
capabilities = Phase0CapabilityService()


def bind_chat_service(chat_service: object | None) -> None:
    capabilities.bind_chat_service(chat_service)


def _timeout(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _authorize(
    *,
    api_key: str | None,
    authorization: str | None,
    nonce: str | None,
    required_scope: str,
    payload: dict,
) -> tuple[CapabilityEnvelope | None, WorkerPrincipal | None]:
    if authorization:
        try:
            principal = authorize_worker_request(
                authorization=authorization,
                nonce=nonce,
                required_scope=required_scope,
                payload=payload,
            )
        except WorkerAuthorizationError as exc:
            return (
                capabilities.error(
                    exc.code,
                    str(exc),
                    request_id=f"req_{uuid.uuid4().hex}",
                ),
                None,
            )
        return None, principal
    expected = os.getenv("MEDAGENT_INTERNAL_API_KEY", "").strip()
    if allow_legacy_internal_key() and expected and api_key and api_key == expected:
        return None, None
    return (
        capabilities.error(
            "UNAUTHORIZED",
            "内部能力凭据无效",
            request_id=f"req_{uuid.uuid4().hex}",
        ),
        None,
    )


def _audit(
    principal: WorkerPrincipal | None,
    *,
    payload: dict,
    operation: str,
    result: CapabilityEnvelope,
) -> None:
    if principal is None:
        return
    data = result.data if isinstance(result.data, dict) else {}
    evidence = data.get("evidence")
    phase1_repository.record_capability_event(
        user_id=principal.user_id,
        session_id=str(payload["session_id"]),
        turn_id=str(payload["turn_id"]),
        action=operation,
        outcome=result.status,
        request_id=result.request_id,
        details={
            "idempotency_key": payload.get("idempotency_key"),
            "metrics": result.metrics,
            "error_code": result.error.code if result.error else None,
        },
        evidence=evidence if isinstance(evidence, list) else None,
    )


def _response(
    envelope: CapabilityEnvelope, *, unauthorized: bool = False
) -> JSONResponse:
    return JSONResponse(
        status_code=401 if unauthorized else 200,
        content=envelope.model_dump(mode="json"),
    )


@router.post("/safety/input-check", response_model=CapabilityEnvelope)
async def input_check(
    payload: InputCheckRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_request_nonce: Annotated[str | None, Header()] = None,
):
    payload_data = payload.model_dump(mode="json")
    denied, principal = _authorize(
        api_key=x_internal_api_key,
        authorization=authorization,
        nonce=x_request_nonce,
        required_scope="safety:input",
        payload=payload_data,
    )
    if denied:
        return _response(denied, unauthorized=True)
    result = await capabilities.invoke(
        operation="safety.input-check",
        idempotency_key=payload.idempotency_key,
        payload=payload.model_dump(),
        timeout_ms=_timeout("MEDAGENT_INPUT_SAFETY_TIMEOUT_MS", 400),
        function=lambda request_id: capabilities.input_check(
            payload.text, request_id
        ),
    )
    _audit(principal, payload=payload_data, operation="safety.input-check", result=result)
    return _response(result)


@router.post("/safety/output-check", response_model=CapabilityEnvelope)
async def output_check(
    payload: OutputCheckRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_request_nonce: Annotated[str | None, Header()] = None,
):
    payload_data = payload.model_dump(mode="json")
    denied, principal = _authorize(
        api_key=x_internal_api_key,
        authorization=authorization,
        nonce=x_request_nonce,
        required_scope="safety:output",
        payload=payload_data,
    )
    if denied:
        return _response(denied, unauthorized=True)
    result = await capabilities.invoke(
        operation="safety.output-check",
        idempotency_key=payload.idempotency_key,
        payload=payload.model_dump(),
        timeout_ms=_timeout("MEDAGENT_OUTPUT_SAFETY_TIMEOUT_MS", 400),
        function=lambda request_id: capabilities.output_check(
            payload.text,
            [item.model_dump() for item in payload.evidence],
            request_id,
        ),
    )
    _audit(principal, payload=payload_data, operation="safety.output-check", result=result)
    return _response(result)


@router.post("/medical/retrieve", response_model=CapabilityEnvelope)
async def medical_retrieve(
    payload: MedicalRetrieveRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_request_nonce: Annotated[str | None, Header()] = None,
):
    payload_data = payload.model_dump(mode="json")
    denied, principal = _authorize(
        api_key=x_internal_api_key,
        authorization=authorization,
        nonce=x_request_nonce,
        required_scope="medical:retrieve",
        payload=payload_data,
    )
    if denied:
        return _response(denied, unauthorized=True)
    result = await capabilities.invoke(
        operation="medical.retrieve",
        idempotency_key=payload.idempotency_key,
        payload=payload.model_dump(),
        timeout_ms=_timeout("MEDAGENT_MEDICAL_RETRIEVAL_TIMEOUT_MS", 1500),
        function=lambda request_id: capabilities.retrieve_medical(
            query=payload.query,
            top_k=payload.top_k,
            department=payload.department,
            turn_id=payload.turn_id,
            request_id=request_id,
        ),
    )
    _audit(principal, payload=payload_data, operation="medical.retrieve", result=result)
    return _response(result)


@router.post("/medical/tools/execute", response_model=CapabilityEnvelope)
async def medical_tool_execute(
    payload: MedicalToolRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_request_nonce: Annotated[str | None, Header()] = None,
):
    payload_data = payload.model_dump(mode="json")
    denied, principal = _authorize(
        api_key=x_internal_api_key,
        authorization=authorization,
        nonce=x_request_nonce,
        required_scope="medical:tools",
        payload=payload_data,
    )
    if denied:
        return _response(denied, unauthorized=True)
    result = await capabilities.invoke(
        operation=f"medical.tools.{payload.tool_name}",
        idempotency_key=payload.idempotency_key,
        payload=payload.model_dump(),
        timeout_ms=_timeout("MEDAGENT_MEDICAL_TOOL_TIMEOUT_MS", 500),
        function=lambda request_id: capabilities.execute_tool(
            payload.tool_name, payload.arguments, request_id
        ),
    )
    _audit(
        principal,
        payload=payload_data,
        operation=f"medical.tools.{payload.tool_name}",
        result=result,
    )
    return _response(result)


@router.post(
    "/voice/sessions/{session_id}/turns",
    response_model=CapabilityEnvelope,
)
async def record_voice_turn(
    session_id: str,
    payload: VoiceTurnRecordRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_request_nonce: Annotated[str | None, Header()] = None,
):
    """把消息、证据与安全检查结果原子关联到同一 turn_id。"""

    payload_data = payload.model_dump(mode="json")
    if session_id != payload.session_id:
        return _response(
            capabilities.error(
                "SESSION_BINDING_MISMATCH",
                "路径 session_id 与请求绑定不一致",
                request_id=f"req_{uuid.uuid4().hex}",
            ),
            unauthorized=True,
        )
    denied, principal = _authorize(
        api_key=x_internal_api_key,
        authorization=authorization,
        nonce=x_request_nonce,
        required_scope="voice:turn:write",
        payload=payload_data,
    )
    if denied:
        return _response(denied, unauthorized=True)
    if principal is None:
        return _response(
            capabilities.error(
                "FORBIDDEN",
                "迁移期 API key 不允许写入受保护语音记录",
                request_id=f"req_{uuid.uuid4().hex}",
            ),
            unauthorized=True,
        )
    async def _write_turn(request_id: str) -> dict:
        await asyncio.to_thread(
            phase1_repository.record_voice_turn,
            user_id=principal.user_id,
            session_id=session_id,
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
            user_text=payload.user_text,
            raw_model_text=payload.raw_model_text,
            final_text=payload.final_text,
            safety_result={
                "input": payload.input_safety,
                "output": payload.output_safety,
                "request_id": request_id,
            },
            evidence=[
                item.model_dump(mode="json") for item in payload.evidence
            ],
        )
        return {"session_id": session_id, "turn_id": payload.turn_id}

    result = await capabilities.invoke(
        operation="voice.turn.write",
        idempotency_key=payload.idempotency_key,
        payload=payload.model_dump(),
        timeout_ms=_timeout("MEDAGENT_VOICE_TURN_WRITE_TIMEOUT_MS", 1000),
        function=_write_turn,
    )
    return _response(result)


@router.post(
    "/voice/sessions/{session_id}/finalize",
    response_model=CapabilityEnvelope,
)
async def finalize_voice_session(
    session_id: str,
    payload: VoiceSessionFinalizeRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_request_nonce: Annotated[str | None, Header()] = None,
):
    """Finalize one summary version and its controlled fact candidates once."""

    payload_data = payload.model_dump(mode="json")
    if session_id != payload.session_id:
        return _response(
            capabilities.error(
                "SESSION_BINDING_MISMATCH",
                "路径 session_id 与请求绑定不一致",
                request_id=f"req_{uuid.uuid4().hex}",
            ),
            unauthorized=True,
        )
    denied, principal = _authorize(
        api_key=x_internal_api_key,
        authorization=authorization,
        nonce=x_request_nonce,
        required_scope="voice:turn:write",
        payload=payload_data,
    )
    if denied:
        return _response(denied, unauthorized=True)
    if principal is None:
        return _response(
            capabilities.error(
                "FORBIDDEN",
                "迁移期 API key 不允许结束受保护语音会话",
                request_id=f"req_{uuid.uuid4().hex}",
            ),
            unauthorized=True,
        )

    async def _finalize(request_id: str) -> dict:
        del request_id
        return await asyncio.to_thread(
            phase1_repository.finalize_voice_session_memory,
            user_id=principal.user_id,
            session_id=session_id,
            summary_version=payload.summary_version,
        )

    result = await capabilities.invoke(
        operation="voice.session.finalize",
        idempotency_key=payload.idempotency_key,
        payload=payload.model_dump(),
        timeout_ms=_timeout("MEDAGENT_VOICE_FINALIZE_TIMEOUT_MS", 5000),
        function=_finalize,
    )
    return _response(result)


@router.post(
    "/voice/sessions/{session_id}/memory-context",
    response_model=CapabilityEnvelope,
)
async def get_voice_memory_context(
    session_id: str,
    payload: VoiceMemoryContextRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_request_nonce: Annotated[str | None, Header()] = None,
):
    """Return a fresh, minimal view of confirmed facts for this bound worker."""

    payload_data = payload.model_dump(mode="json")
    if session_id != payload.session_id:
        return _response(
            capabilities.error(
                "SESSION_BINDING_MISMATCH",
                "路径 session_id 与请求绑定不一致",
                request_id=f"req_{uuid.uuid4().hex}",
            ),
            unauthorized=True,
        )
    denied, principal = _authorize(
        api_key=x_internal_api_key,
        authorization=authorization,
        nonce=x_request_nonce,
        required_scope="voice:memory:read",
        payload=payload_data,
    )
    if denied:
        return _response(denied, unauthorized=True)
    if principal is None:
        return _response(
            capabilities.error(
                "FORBIDDEN",
                "迁移期 API key 不允许读取受控记忆",
                request_id=f"req_{uuid.uuid4().hex}",
            ),
            unauthorized=True,
        )
    items = await asyncio.to_thread(
        phase1_repository.list_medical_fact_memories,
        user_id=principal.user_id,
        statuses=["confirmed"],
    )
    envelope = CapabilityEnvelope(
        request_id=f"req_{uuid.uuid4().hex}",
        status="ok",
        data={
            # Minimal disclosure: no source text, document id, confidence,
            # structured value, timestamps, or superseded/rejected versions.
            "confirmed_facts": [
                " ".join(str(item["content"]).split())[:500]
                for item in items[:50]
                if str(item.get("content") or "").strip()
            ],
            "fresh": True,
        },
        metrics={"memory_count": min(len(items), 50)},
    )
    return _response(envelope)
