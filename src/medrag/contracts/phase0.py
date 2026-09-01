"""阶段 0 内部能力 API 的冻结数据契约。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

API_VERSION = "internal/v1"
ERROR_CODES = frozenset({
    "INVALID_REQUEST", "UNAUTHORIZED", "IDEMPOTENCY_CONFLICT",
    "CAPABILITY_TIMEOUT", "CAPABILITY_UNAVAILABLE", "UNSUPPORTED_TOOL",
    "TOOL_EXECUTION_FAILED",
})
VOICE_TOOL_NAMES = frozenset({
    "search_medical_knowledge", "search_personal_knowledge_base",
    "calculate_dosage", "guide_department", "lookup_normal_range",
})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CapabilityContext(BaseModel):
    """调用关联信息；不接受或信任 worker 自报的用户身份。"""

    session_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=200)


class Evidence(BaseModel):
    evidence_id: str
    turn_id: str
    source_type: Literal["medical", "personal"]
    fact_type: str = "reference"
    subject_scope: Literal["user_specific", "general"] = "general"
    source_category: str
    source_id: str
    document_id: str = ""
    title: str = ""
    content_preview: str = ""
    authority_level: str = "unknown"
    verification_status: str = "unverified"
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    version: str = "1"
    score: float | None = None
    confidence: float | None = None
    request_id: str
    latency_ms: float
    created_at: datetime = Field(default_factory=utc_now)


class CapabilityError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class CapabilityEnvelope(BaseModel):
    request_id: str
    status: Literal["ok", "error"]
    data: Any = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: CapabilityError | None = None


class InputCheckRequest(CapabilityContext):
    text: str = Field(min_length=1, max_length=4000)


class OutputCheckRequest(CapabilityContext):
    text: str = Field(min_length=1, max_length=8000)
    evidence: list[Evidence] = Field(default_factory=list)


class MedicalRetrieveRequest(CapabilityContext):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    department: str | None = Field(default=None, max_length=100)


class MedicalToolRequest(CapabilityContext):
    tool_name: Literal["calculate_dosage", "guide_department", "lookup_normal_range"]
    arguments: dict[str, Any] = Field(default_factory=dict)
