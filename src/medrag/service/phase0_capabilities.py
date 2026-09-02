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

from medcontracts.phase0 import CapabilityEnvelope, Evidence
from medrag.rag.safety_guard import RED_SIGNALS
from medrag.tools import get_tool_registry

EMERGENCY_RESPONSE = "你描述的情况可能存在紧急风险。请立即拨打120或前往最近的急诊科，并尽量让身边的人陪同；不要自行驾车或等待在线回复。"
CLARIFY_RESPONSE = "请确认：这是你本人现在正在发生的症状吗？如果症状正在发生或迅速加重，请立即拨打120或前往急诊。"
OUTPUT_FALLBACK = "这段内容未通过医疗安全检查，暂不播报。涉及诊断、用药或急症时，请咨询医生；如有紧急症状请立即拨打120。"
CONFLICT_NOTICE = "检索到的依据存在无法可靠消解的冲突，我不会替你选边。请核对下列来源，并由医生结合你的实际情况确认。"

_RED_PATTERN = "|".join(map(re.escape, RED_SIGNALS))
_CURRENT_SELF = re.compile(
    r"(我|本人|自己).{0,8}(现在|正在|此刻|刚刚|突然|目前)|"
    r"(?:现在|正在|此刻|刚刚|突然|目前).{0,8}(我|本人|自己)"
)
_SELF_PERSON = re.compile(r"(我|本人|自己)")
_CURRENT_MARKER = re.compile(r"(现在|正在|此刻|刚刚|突然|目前)")
_NEGATED = re.compile(
    rf"(没有|并无|不是|未出现|不再|否认).{{0,8}}?({_RED_PATTERN})"
)
_HISTORICAL = re.compile(r"(以前|过去|去年|曾经|历史上|小时候|已经好了)")
_OTHER_PERSON = re.compile(rf"(他|她|家人|朋友|孩子|老人|患者).{{0,10}}({_RED_PATTERN})")
_EDUCATIONAL = re.compile(r"(是什么|什么意思|有哪些|为什么|科普|如何定义|会不会|可能吗)")
_ONGOING = re.compile(r"(还在|仍然|持续|一直|越来越|没有缓解|未缓解)")
_SPECIAL_POPULATIONS = {
    "child": re.compile(r"(儿童|孩子|小孩|婴儿|宝宝|未成年)"),
    "pregnant": re.compile(r"(孕妇|怀孕|孕期|妊娠|哺乳期)"),
    "older_adult": re.compile(r"(老人|老年人|高龄|七十岁|八十岁|九十岁)"),
}
_CLAUSE_BOUNDARY = re.compile(r"[，,。！？!?；;\n]+|但是|但|不过|然而")
_DANGEROUS_OUTPUT = (
    re.compile(r"(肯定|一定|确诊|就是).{0,10}(癌|肿瘤|心梗|脑梗|疾病|病)"),
    re.compile(r"(加倍|大量|超量).{0,8}(服用|吃|注射)"),
    re.compile(r"(不用|不必).{0,8}(就医|急诊|拨打120)"),
    re.compile(r"(过敏|禁忌|孕妇|妊娠).{0,10}(也可以|照常|随便).{0,6}(服用|使用|吃)"),
    re.compile(r"(胸痛|呼吸困难|昏迷|抽搐|大出血).{0,12}(先观察|等一等|睡一觉)"),
)
_DOSAGE_CLAIM = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|毫克|g|克|ml|毫升|片|粒)")
_OVERSIGHT_NOTICE = re.compile(r"(遵医嘱|医生|药师|处方|说明书)")
DOSAGE_NOTICE = "具体剂量必须以医生处方、药师核对和药品说明书为准，不要自行调整。"


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
        negated = bool(_NEGATED.search(text))
        active_text = _NEGATED.sub("", text)
        red_types = [
            label
            for keyword, label in RED_SIGNALS.items()
            if keyword in active_text
        ]
        semantic_context = {
            "current_self": False,
            "current_other": False,
            "other_person": bool(_OTHER_PERSON.search(text)),
            "historical": bool(_HISTORICAL.search(text)),
            "negated": negated,
            "educational": bool(_EDUCATIONAL.search(text)),
            "ongoing": bool(_ONGOING.search(text)),
            "special_populations": [
                name
                for name, pattern in _SPECIAL_POPULATIONS.items()
                if pattern.search(text)
            ],
        }
        if not red_types:
            return {
                "action": "allow",
                "risk_level": "none",
                "risk_types": [],
                "fixed_response": None,
                "semantic_context": semantic_context,
            }
        red_clauses = [
            clause
            for clause in _CLAUSE_BOUNDARY.split(active_text)
            if any(keyword in clause for keyword in RED_SIGNALS)
        ]
        semantic_context["current_self"] = any(
            (
                _CURRENT_SELF.search(clause)
                or (
                    _SELF_PERSON.search(clause)
                    and _ONGOING.search(clause)
                    and not _EDUCATIONAL.search(clause)
                )
            )
            and not _OTHER_PERSON.search(clause)
            for clause in red_clauses
        )
        semantic_context["current_other"] = any(
            _OTHER_PERSON.search(clause)
            and _CURRENT_MARKER.search(clause)
            and not _EDUCATIONAL.search(clause)
            for clause in red_clauses
        )
        if (
            semantic_context["current_self"]
            or semantic_context["current_other"]
        ):
            return {
                "action": "emergency",
                "risk_level": "red",
                "risk_types": red_types,
                "fixed_response": EMERGENCY_RESPONSE,
                "semantic_context": semantic_context,
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
                "semantic_context": semantic_context,
            }
        if _EDUCATIONAL.search(text):
            return {
                "action": "allow",
                "risk_level": "educational",
                "risk_types": red_types,
                "fixed_response": None,
                "semantic_context": semantic_context,
            }
        return {
            "action": "clarify",
            "risk_level": "uncertain",
            "risk_types": red_types,
            "fixed_response": CLARIFY_RESPONSE,
            "semantic_context": semantic_context,
        }

    async def output_check(
        self, text: str, evidence: list[dict[str, Any]], request_id: str
    ) -> dict[str, Any]:
        del request_id
        violations = [
            pattern.pattern for pattern in _DANGEROUS_OUTPUT if pattern.search(text)
        ]
        conflicts = analyze_evidence_conflicts(evidence)
        warnings = [
            {
                "code": "PERSONAL_GENERAL_NOT_MEDICAL_FACT",
                "evidence_id": str(item.get("evidence_id") or ""),
            }
            for item in evidence
            if item.get("source_type") == "personal"
            and item.get("subject_scope") == "general"
        ]
        high_risk_conflicts = [
            item for item in conflicts if item.get("high_risk")
        ]
        if high_risk_conflicts:
            return {
                "allowed": False,
                "safe_text": CONFLICT_NOTICE,
                "violations": ["unresolved_high_risk_evidence_conflict"],
                "rule_version": "phase2.1",
                "evidence_conflicts": conflicts,
                "evidence_warnings": warnings,
                "conflict_notice": CONFLICT_NOTICE,
            }
        if violations:
            return {
                "allowed": False,
                "safe_text": (
                    f"{OUTPUT_FALLBACK}\n{CONFLICT_NOTICE}"
                    if conflicts
                    else OUTPUT_FALLBACK
                ),
                "violations": violations,
                "rule_version": "phase2.1",
                "evidence_conflicts": conflicts,
                "evidence_warnings": warnings,
                "conflict_notice": CONFLICT_NOTICE if conflicts else None,
            }
        required_notices = []
        safe_text = text
        if _DOSAGE_CLAIM.search(text) and not _OVERSIGHT_NOTICE.search(text):
            required_notices.append(DOSAGE_NOTICE)
            safe_text = f"{text.rstrip()}\n{DOSAGE_NOTICE}"
        return {
            "allowed": True,
            "safe_text": safe_text,
            "violations": [],
            "rule_version": "phase2.1",
            "evidence_count": len(evidence),
            "evidence_conflicts": conflicts,
            "evidence_warnings": warnings,
            "conflict_notice": CONFLICT_NOTICE if conflicts else None,
            "required_notices": required_notices,
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
                        fact_type=str(
                            raw.get("fact_type")
                            or raw.get("type")
                            or "reference"
                        ),
                        fact_subject_id=str(
                            raw.get("fact_subject_id")
                            or raw.get("subject_id")
                            or raw.get("drug_id")
                            or raw.get("drug_name")
                            or raw.get("indicator_id")
                            or raw.get("indicator_name")
                            or raw.get("test_name")
                            or ""
                        ),
                        subject_scope=(
                            str(raw.get("subject_scope"))
                            if raw.get("subject_scope")
                            in {"user_specific", "general"}
                            else "general"
                        ),
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


def analyze_evidence_conflicts(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按证据属性发现冲突；不以 medical/personal 来源硬编码选边。"""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in evidence:
        fact_type = str(item.get("fact_type") or "reference").strip().casefold()
        if fact_type in {"", "reference", "unknown"}:
            continue
        fact_subject_id = _fact_subject_key(item, fact_type)
        if not fact_subject_id:
            continue
        value = str(
            item.get("structured_value")
            or item.get("claim_value")
            or item.get("content_preview")
            or ""
        ).strip()
        if not value:
            continue
        groups.setdefault((fact_type, fact_subject_id), []).append(
            {**item, "_value": value}
        )

    conflicts: list[dict[str, Any]] = []
    high_risk_terms = ("diagnos", "处方", "prescription", "dosage", "剂量", "禁忌", "contraind")
    for (fact_type, fact_subject_id), items in groups.items():
        values = {
            re.sub(r"\s+", "", str(item["_value"])).casefold()
            for item in items
        }
        if len(items) < 2 or len(values) < 2:
            continue
        conflicts.append(
            {
                "fact_type": fact_type,
                "fact_subject_id": fact_subject_id,
                "resolution": "unresolved",
                "high_risk": any(term in fact_type for term in high_risk_terms),
                "reason": "同一事实类型存在不同陈述，需结合用户特异性、时间、版本、权威性和有效状态核实",
                "sources": [
                    {
                        "evidence_id": str(item.get("evidence_id") or ""),
                        "source_type": str(item.get("source_type") or ""),
                        "subject_scope": str(item.get("subject_scope") or ""),
                        "source_category": str(item.get("source_category") or ""),
                        "title": str(item.get("title") or ""),
                        "content_preview": str(item.get("content_preview") or ""),
                        "authority_level": str(item.get("authority_level") or "unknown"),
                        "verification_status": str(item.get("verification_status") or "unverified"),
                        "observed_at": item.get("observed_at"),
                        "valid_from": item.get("valid_from"),
                        "valid_to": item.get("valid_to"),
                        "version": str(item.get("version") or "1"),
                        "confidence": item.get("confidence"),
                    }
                    for item in items
                ],
            }
        )
    return conflicts


def _fact_subject_key(item: dict[str, Any], fact_type: str) -> str:
    """返回稳定事实主体；主体不明确时不把不同对象误判为冲突。"""

    explicit = str(
        item.get("fact_subject_id")
        or item.get("subject_id")
        or item.get("drug_id")
        or item.get("drug_name")
        or item.get("indicator_id")
        or item.get("indicator_name")
        or item.get("test_name")
        or ""
    ).strip()
    if explicit:
        return re.sub(r"\s+", "", explicit).casefold()
    if not any(term in fact_type for term in ("dosage", "剂量", "range", "范围")):
        return ""
    preview = str(item.get("content_preview") or "").strip()
    match = re.search(
        r"([A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff_-]{0,39})"
        r"\s*[:：]?\s*\d+(?:\.\d+)?\s*(?:mg|毫克|g|克|ml|毫升|片|粒)",
        preview,
        re.IGNORECASE,
    )
    if not match:
        return ""
    inferred = match.group(1).casefold()
    if inferred in {"每日", "每次", "剂量", "参考", "正常", "范围"}:
        return ""
    return inferred
