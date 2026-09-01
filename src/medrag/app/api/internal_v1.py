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

router = APIRouter()
capabilities = Phase0CapabilityService()


def bind_chat_service(chat_service: object | None) -> None:
    capabilities.bind_chat_service(chat_service)


def _timeout(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _authorize(api_key: str | None) -> CapabilityEnvelope | None:
    expected = os.getenv("MEDAGENT_INTERNAL_API_KEY", "").strip()
    if not expected or not api_key or api_key != expected:
        return capabilities.error(
            "UNAUTHORIZED",
            "内部能力凭据无效",
            request_id=f"req_{uuid.uuid4().hex}",
        )
    return None


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
):
    denied = _authorize(x_internal_api_key)
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
    return _response(result)


@router.post("/safety/output-check", response_model=CapabilityEnvelope)
async def output_check(
    payload: OutputCheckRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
):
    denied = _authorize(x_internal_api_key)
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
    return _response(result)


@router.post("/medical/retrieve", response_model=CapabilityEnvelope)
async def medical_retrieve(
    payload: MedicalRetrieveRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
):
    denied = _authorize(x_internal_api_key)
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
    return _response(result)


@router.post("/medical/tools/execute", response_model=CapabilityEnvelope)
async def medical_tool_execute(
    payload: MedicalToolRequest,
    x_internal_api_key: Annotated[str | None, Header()] = None,
):
    denied = _authorize(x_internal_api_key)
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
    return _response(result)
