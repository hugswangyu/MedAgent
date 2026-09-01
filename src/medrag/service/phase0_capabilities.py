"""阶段 0 retrieval-only、安全检查和确定性医疗工具能力。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Awaitable, Callable

from medrag.contracts.phase0 import CapabilityEnvelope, Evidence
from medrag.rag.safety_guard import RED_SIGNALS
from medrag.tools import get_tool_registry

EMERGENCY_RESPONSE = "你描述的情况可能存在紧急风险。请立即拨打120或前往最近的急诊科，并尽量让身边的人陪同；不要自行驾车或等待在线回复。"
CLARIFY_RESPONSE = "请确认：这是你本人现在正在发生的症状吗？如果症状正在发生或迅速加重，请立即拨打120或前往急诊。"
OUTPUT_FALLBACK = "这段内容未通过医疗安全检查，暂不播报。涉及诊断、用药或急症时，请咨询医生；如有紧急症状请立即拨打120。"

_RED_PATTERN = "|".join(map(re.escape, RED_SIGNALS))
_CURRENT_SELF = re.compile(
    r"(我|本人|自己).{0,8}(现在|正在|此刻|刚刚|突然|目前)|"
    r"(?:现在|正在|此刻|刚刚|突然|目前).{0,8}(我|本人|自己)"
)
_NEGATED = re.compile(
    rf"(没有|并无|不是|未出现|不再|否认).{{0,8}}?({_RED_PATTERN})"
)
_HISTORICAL = re.compile(r"(以前|过去|去年|曾经|历史上|小时候|已经好了)")
_OTHER_PERSON = re.compile(rf"(他|她|家人|朋友|孩子|老人|患者).{{0,10}}({_RED_PATTERN})")
_EDUCATIONAL = re.compile(r"(是什么|什么意思|有哪些|为什么|科普|如何定义|会不会|可能吗)")
_CLAUSE_BOUNDARY = re.compile(r"[，,。！？!?；;\n]+|但是|但|不过|然而")
_DANGEROUS_OUTPUT = (
    re.compile(r"(肯定|一定|确诊|就是).{0,10}(癌|肿瘤|心梗|脑梗|疾病|病)"),
    re.compile(r"(加倍|大量|超量).{0,8}(服用|吃|注射)"),
    re.compile(r"(不用|不必).{0,8}(就医|急诊|拨打120)"),
)


class Phase0CapabilityService:
    """把现有组件收敛为无生成、可幂等调用的内部能力。"""

    def __init__(self, chat_service: Any | None = None, *, cache_size: int = 512) -> None:
        self.chat_service = chat_service
        self.tool_registry = get_tool_registry()
        self._cache_size = cache_size
        self._idempotency_cache: OrderedDict[
            str, tuple[str, CapabilityEnvelope]
        ] = OrderedDict()
        self._inflight: dict[
            str, tuple[str, asyncio.Future[CapabilityEnvelope]]
        ] = {}
        self._lock = asyncio.Lock()

    def bind_chat_service(self, chat_service: Any | None) -> None:
        self.chat_service = chat_service

    async def invoke(
        self,
        *,
        operation: str,
        idempotency_key: str,
        payload: dict[str, Any],
        timeout_ms: int,
        function: Callable[[str], Awaitable[dict[str, Any]]],
    ) -> CapabilityEnvelope:
        cache_key = f"{operation}:{idempotency_key}"
        fingerprint = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()
        owner = False
        async with self._lock:
            cached = self._idempotency_cache.get(cache_key)
            if cached:
                if cached[0] != fingerprint:
                    return self.error(
                        "IDEMPOTENCY_CONFLICT",
                        "同一幂等键不能用于不同请求",
                        request_id=f"req_{uuid.uuid4().hex}",
                    )
                self._idempotency_cache.move_to_end(cache_key)
                result = deepcopy(cached[1])
                result.metrics = {**result.metrics, "idempotency_replay": True}
                return result
            inflight = self._inflight.get(cache_key)
            if inflight:
                if inflight[0] != fingerprint:
                    return self.error(
                        "IDEMPOTENCY_CONFLICT",
                        "同一幂等键不能用于不同请求",
                        request_id=f"req_{uuid.uuid4().hex}",
                    )
                future = inflight[1]
            else:
                future = asyncio.get_running_loop().create_future()
                self._inflight[cache_key] = (fingerprint, future)
                owner = True

        if not owner:
            result = deepcopy(await asyncio.shield(future))
            result.metrics = {
                **result.metrics,
                "idempotency_replay": True,
                "idempotency_waited": True,
            }
            return result

        request_id = f"req_{uuid.uuid4().hex}"
        started = time.perf_counter()
        try:
            data = await asyncio.wait_for(
                function(request_id), timeout=timeout_ms / 1000.0
            )
            result = CapabilityEnvelope(
                request_id=request_id,
                status="ok",
                data=data,
                metrics={
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000, 1
                    ),
                    "timeout_ms": timeout_ms,
                },
            )
        except TimeoutError:
            result = self.error(
                "CAPABILITY_TIMEOUT",
                f"{operation} 超过 {timeout_ms}ms 硬超时",
                request_id=request_id,
                retryable=True,
                metrics={
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000, 1
                    ),
                    "timeout_ms": timeout_ms,
                },
            )
        except Exception as exc:
            code = (
                "UNSUPPORTED_TOOL"
                if str(exc) == "UNSUPPORTED_TOOL"
                else "CAPABILITY_UNAVAILABLE"
            )
            result = self.error(
                code,
                f"{operation} 暂不可用",
                request_id=request_id,
                retryable=code == "CAPABILITY_UNAVAILABLE",
                details={"type": type(exc).__name__},
                metrics={
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000, 1
                    ),
                    "timeout_ms": timeout_ms,
                },
            )

        async with self._lock:
            self._idempotency_cache[cache_key] = (fingerprint, deepcopy(result))
            while len(self._idempotency_cache) > self._cache_size:
                self._idempotency_cache.popitem(last=False)
            self._inflight.pop(cache_key, None)
            if not future.done():
                future.set_result(deepcopy(result))
        return result

    async def input_check(self, text: str, request_id: str) -> dict[str, Any]:
        del request_id
        active_text = _NEGATED.sub("", text)
        red_types = [
            label
            for keyword, label in RED_SIGNALS.items()
            if keyword in active_text
        ]
        if not red_types:
            return {
                "action": "allow",
                "risk_level": "none",
                "risk_types": [],
                "fixed_response": None,
            }
        red_clauses = [
            clause
            for clause in _CLAUSE_BOUNDARY.split(active_text)
            if any(keyword in clause for keyword in RED_SIGNALS)
        ]
        if any(_CURRENT_SELF.search(clause) for clause in red_clauses):
            return {
                "action": "emergency",
                "risk_level": "red",
                "risk_types": red_types,
                "fixed_response": EMERGENCY_RESPONSE,
            }
        if (
            _HISTORICAL.search(text)
            or _OTHER_PERSON.search(text)
        ):
            return {
                "action": "allow",
                "risk_level": "context_only",
                "risk_types": red_types,
                "fixed_response": None,
            }
        if _EDUCATIONAL.search(text):
            return {
                "action": "allow",
                "risk_level": "educational",
                "risk_types": red_types,
                "fixed_response": None,
            }
        return {
            "action": "clarify",
            "risk_level": "uncertain",
            "risk_types": red_types,
            "fixed_response": CLARIFY_RESPONSE,
        }

    async def output_check(
        self, text: str, evidence: list[dict[str, Any]], request_id: str
    ) -> dict[str, Any]:
        del request_id
        violations = [
            pattern.pattern for pattern in _DANGEROUS_OUTPUT if pattern.search(text)
        ]
        if violations:
            return {
                "allowed": False,
                "safe_text": OUTPUT_FALLBACK,
                "violations": violations,
                "rule_version": "phase0.1",
            }
        return {
            "allowed": True,
            "safe_text": text,
            "violations": [],
            "rule_version": "phase0.1",
            "evidence_count": len(evidence),
        }

    async def retrieve_medical(
        self,
        *,
        query: str,
        top_k: int,
        department: str | None,
        turn_id: str,
        request_id: str,
    ) -> dict[str, Any]:
        if self.chat_service is None:
            raise RuntimeError("MedicalChatService 尚未就绪")
        retriever = self.chat_service.hybrid_retriever
        started = time.perf_counter()
        route = await asyncio.to_thread(
            retriever.router.route, query, False
        )
        route = {**route, "needs_case_context": False}
        result = await asyncio.to_thread(
            retriever.retrieve, query, top_k, department, None, route
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        evidence: list[Evidence] = []
        groups = (
            ("knowledge_graph", result.get("kg_results", [])),
            ("medical_qa", result.get("qa_results", [])),
        )
        for category, items in groups:
            for index, raw in enumerate(items or []):
                if not isinstance(raw, dict):
                    raw = {"content": str(raw)}
                source_id = str(
                    raw.get("id")
                    or raw.get("source")
                    or f"{category}:{index}"
                )
                preview = str(
                    raw.get("content")
                    or raw.get("answer")
                    or raw.get("text")
                    or raw.get("description")
                    or ""
                )[:500]
                score = raw.get(
                    "rerank_score", raw.get("score", raw.get("rrf_score"))
                )
                evidence.append(
                    Evidence(
                        evidence_id=(
                            "ev_"
                            + hashlib.sha256(
                                (
                                    request_id + category + source_id
                                ).encode()
                            ).hexdigest()[:24]
                        ),
                        turn_id=turn_id,
                        source_type="medical",
                        source_category=category,
                        source_id=source_id,
                        document_id=str(
                            raw.get("document_id") or raw.get("doc_id") or ""
                        ),
                        title=str(
                            raw.get("title") or raw.get("question") or ""
                        ),
                        content_preview=preview,
                        authority_level=str(
                            raw.get("authority_level") or "reference_corpus"
                        ),
                        verification_status=str(
                            raw.get("verification_status") or "indexed"
                        ),
                        score=(
                            float(score)
                            if isinstance(score, (int, float))
                            else None
                        ),
                        request_id=request_id,
                        latency_ms=latency_ms,
                    )
                )
        return {
            "query": query,
            "route": result.get("route", route),
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "evidence_count": len(evidence),
            "retrieval_only": True,
        }

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        del request_id
        mapping = {
            "calculate_dosage": "剂量计算",
            "guide_department": "科室导诊",
            "lookup_normal_range": "正常值查询",
        }
        internal_name = mapping.get(tool_name)
        if internal_name is None:
            raise ValueError("UNSUPPORTED_TOOL")
        result = await asyncio.to_thread(
            self.tool_registry.execute, internal_name, **arguments
        )
        if result.startswith("工具「") and "执行失败" in result:
            raise RuntimeError(result)
        return {
            "tool_name": tool_name,
            "result": result,
            "deterministic": True,
        }

    @staticmethod
    def error(
        code: str,
        message: str,
        *,
        request_id: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> CapabilityEnvelope:
        return CapabilityEnvelope(
            request_id=request_id,
            status="error",
            metrics=metrics or {},
            error={
                "code": code,
                "message": message,
                "retryable": retryable,
                "details": details or {},
            },
        )
