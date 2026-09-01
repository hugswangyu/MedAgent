"""阶段 0 内部能力 HTTP API。"""

from __future__ import annotations

import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from medrag.contracts.phase0 import (
    CapabilityEnvelope,
    InputCheckRequest,
    MedicalRetrieveRequest,
    MedicalToolRequest,
    OutputCheckRequest,
)
from medrag.service.phase0_capabilities import Phase0CapabilityService
from medrag.app.worker_auth import (
    WorkerAuthorizationError,
    WorkerPrincipal,
    allow_legacy_internal_key,
    authorize_worker_request,
)
from medrag.infrastructure.storage import phase1_repository

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
