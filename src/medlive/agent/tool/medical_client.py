"""MedAgent 阶段 0 内部能力 HTTP 客户端。"""

from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import aiohttp

from medcontracts.phase0 import CapabilityEnvelope


class MedicalCapabilityError(RuntimeError):
    """内部能力返回错误、超时或不符合冻结契约。"""


@dataclass(slots=True)
class CapabilityResult:
    """统一能力调用结果。"""

    request_id: str
    data: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


class MedicalCapabilityClient:
    """只通过受保护的 /internal/v1 HTTP 契约访问 MedAgent。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        worker_token: str = "",
        worker_token_factory: Callable[[], str | Awaitable[str]] | None = None,
        session_id: str,
        input_timeout_ms: int = 400,
        retrieval_timeout_ms: int = 1500,
        tool_timeout_ms: int = 500,
        output_timeout_ms: int = 400,
    ) -> None:
        if not api_key.strip() and not worker_token.strip() and worker_token_factory is None:
            raise ValueError("必须显式配置 worker token 或迁移期 internal API key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.worker_token = worker_token
        self.worker_token_factory = worker_token_factory
        self.session_id = session_id
        self.input_timeout_ms = input_timeout_ms
        self.retrieval_timeout_ms = retrieval_timeout_ms
        self.tool_timeout_ms = tool_timeout_ms
        self.output_timeout_ms = output_timeout_ms
        self._session: aiohttp.ClientSession | None = None

    async def input_check(
        self, *, text: str, turn_id: str
    ) -> CapabilityResult:
        return await self._post(
            "/internal/v1/safety/input-check",
            {"text": text},
            turn_id=turn_id,
            operation="input-check",
            timeout_ms=self.input_timeout_ms,
        )

    async def output_check(
        self,
        *,
        text: str,
        turn_id: str,
        evidence: list[dict[str, Any]],
    ) -> CapabilityResult:
        return await self._post(
            "/internal/v1/safety/output-check",
            {"text": text, "evidence": evidence},
            turn_id=turn_id,
            operation="output-check",
            timeout_ms=self.output_timeout_ms,
        )

    async def retrieve_medical(
        self,
        *,
        query: str,
        turn_id: str,
        top_k: int = 5,
        department: str | None = None,
    ) -> CapabilityResult:
        return await self._post(
            "/internal/v1/medical/retrieve",
            {"query": query, "top_k": top_k, "department": department},
            turn_id=turn_id,
            operation="medical-retrieve",
            timeout_ms=self.retrieval_timeout_ms,
        )

    async def execute_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        turn_id: str,
    ) -> CapabilityResult:
        return await self._post(
            "/internal/v1/medical/tools/execute",
            {"tool_name": tool_name, "arguments": arguments},
            turn_id=turn_id,
            operation=f"tool-{tool_name}",
            timeout_ms=self.tool_timeout_ms,
        )

    async def record_turn(
        self,
        *,
        turn_id: str,
        turn_index: int,
        user_text: str,
        raw_model_text: str,
        final_text: str,
        input_safety: dict[str, Any],
        output_safety: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> CapabilityResult:
        """按 turn_id 幂等关联消息、证据和安全记录。"""

        return await self._post(
            (
                "/internal/v1/voice/sessions/"
                f"{quote(self.session_id, safe='')}/turns"
            ),
            {
                "turn_index": turn_index,
                "user_text": user_text,
                "raw_model_text": raw_model_text,
                "final_text": final_text,
                "input_safety": input_safety,
                "output_safety": output_safety,
                "evidence": evidence,
            },
            turn_id=turn_id,
            operation="voice-turn-write",
            timeout_ms=max(self.output_timeout_ms, 1000),
        )

    async def finalize_session(self, *, summary_version: int = 1) -> CapabilityResult:
        """Close the PostgreSQL summary/memory loop idempotently."""

        return await self._post(
            (
                "/internal/v1/voice/sessions/"
                f"{quote(self.session_id, safe='')}/finalize"
            ),
            {"summary_version": summary_version},
            turn_id="session-finalize",
            operation=f"voice-session-finalize-v{summary_version}",
            timeout_ms=5000,
        )

    async def confirmed_memory_context(self, *, turn_id: str) -> CapabilityResult:
        """Fetch a fresh minimal confirmed-memory view for one turn."""

        return await self._post(
            (
                "/internal/v1/voice/sessions/"
                f"{quote(self.session_id, safe='')}/memory-context"
            ),
            {},
            turn_id=turn_id,
            operation="voice-memory-context",
            timeout_ms=1000,
        )

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        turn_id: str,
        operation: str,
        timeout_ms: int,
    ) -> CapabilityResult:
        payload = {
            "session_id": self.session_id,
            "turn_id": turn_id,
            "idempotency_key": self._idempotency_key(
                operation=operation, turn_id=turn_id, body=body
            ),
            **body,
        }
        headers = {"Content-Type": "application/json"}
        token_or_awaitable = (
            self.worker_token_factory()
            if self.worker_token_factory is not None
            else self.worker_token
        )
        token = (
            await token_or_awaitable
            if inspect.isawaitable(token_or_awaitable)
            else token_or_awaitable
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-Request-Nonce"] = str(uuid.uuid4())
        elif self.api_key:
            headers["X-Internal-API-Key"] = self.api_key
        timeout = aiohttp.ClientTimeout(
            total=max(timeout_ms, 1) / 1000.0
        )
        try:
            session = self._session
            if session is None or session.closed:
                session = aiohttp.ClientSession()
                self._session = session
            async with session.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                raw = await response.json(content_type=None)
        except Exception as exc:
            raise MedicalCapabilityError(
                f"{operation} HTTP 调用失败: {type(exc).__name__}"
            ) from exc

        if not isinstance(raw, dict):
            raise MedicalCapabilityError(f"{operation} 返回非对象响应")
        try:
            envelope = CapabilityEnvelope.model_validate(raw)
        except ValueError as exc:
            raise MedicalCapabilityError(
                f"{operation} 响应不符合 medcontracts 冻结契约"
            ) from exc
        if envelope.status != "ok":
            code = envelope.error.code if envelope.error else "CAPABILITY_UNAVAILABLE"
            message = envelope.error.message if envelope.error else operation
            raise MedicalCapabilityError(f"{code}: {message}")
        if not isinstance(envelope.data, dict):
            raise MedicalCapabilityError(
                f"{operation} 响应缺少冻结契约字段"
            )
        return CapabilityResult(
            request_id=envelope.request_id,
            data=envelope.data,
            metrics=envelope.metrics,
        )

    async def aclose(self) -> None:
        """关闭复用的 HTTP 连接池。"""

        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _idempotency_key(
        self, *, operation: str, turn_id: str, body: dict[str, Any]
    ) -> str:
        canonical = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return f"{self.session_id}:{turn_id}:{operation}:{digest}"
