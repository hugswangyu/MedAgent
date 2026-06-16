"""src/medrag/harness/types.py — MedAgent 医疗场景核心类型。

不是 AGI-saber 的机械翻译，而是围绕医疗问答场景设计：
  - 证据溯源（每段回答来自哪里）
  - 置信度（医疗回答不可靠时明确告知）
  - 安全检测（风险检测是强制阶段）
  - 渐进式降级（失败时给出用户可理解的提示）
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =========================================================================
# 工具执行状态
# =========================================================================


class ToolStatus(enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"    # 部分成功（如单路检索成功）
    TIMEOUT = "timeout"
    SKIPPED = "skipped"      # 竞速被跳过
    CANCELLED = "cancelled"


# =========================================================================
# 医疗置信度
# =========================================================================


class Confidence(str, enum.Enum):
    """医疗回答的置信度等级。

    直接影响 SafetyGuard 是否追加免责声明。
    """
    HIGH = "high"          # 多源确认（KG + QA + Case）
    MEDIUM = "medium"      # 单源确认（仅 QA 或仅 KG）
    LOW = "low"            # 无检索结果，仅 LLM 自身知识
    NONE = "none"          # 生成失败或全部降级


# =========================================================================
# 证据溯源
# =========================================================================


@dataclass
class SourceTrace:
    """单条证据的来源追溯。

    用于在回答中标注 ``【来源：知识图谱】``，提升医疗可信度和可审计性。
    """
    source_type: str         # "kg" | "qa" | "es" | "case" | "llm" | "cache"
    content: str             # 原始文本片段
    score: float = 0.0       # RRF / rerank 分数
    source_name: str = ""    # 如 "neo4j_medical_kg", "milvus_cmedqa2"
    rank: int = 0            # 在结果中的排序

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "content": self.content[:200],
            "score": self.score,
            "source_name": self.source_name,
            "rank": self.rank,
        }


# =========================================================================
# 医疗工具执行结果
# =========================================================================


@dataclass
class MedToolResult:
    """统一的医疗工具执行结果。

    相比通用 ExecResult，增加了医疗场景特有的：
      - sources[]: 证据溯源列表
      - confidence: 置信度等级
      - user_facing_warning: 面向用户的降级提示（中文）
      - risk_flags: 触发的风险关键词
    """
    status: ToolStatus
    tool_name: str
    data: Any = None
    error: Optional[str] = None
    warning: Optional[str] = None           # 内部日志用
    user_facing_warning: Optional[str] = None  # 展示给用户
    latency_ms: float = 0.0
    retry_count: int = 0
    sources: List[SourceTrace] = field(default_factory=list)
    confidence: Confidence = Confidence.HIGH
    risk_flags: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, tool_name: str, data: Any,
           sources: Optional[List[SourceTrace]] = None,
           confidence: Confidence = Confidence.HIGH,
           latency_ms: float = 0.0) -> MedToolResult:
        return cls(
            status=ToolStatus.SUCCESS,
            tool_name=tool_name,
            data=data,
            sources=sources or [],
            confidence=confidence,
            latency_ms=latency_ms,
        )

    @classmethod
    def failed(cls, tool_name: str, error: str,
               user_facing_warning: Optional[str] = None,
               latency_ms: float = 0.0) -> MedToolResult:
        return cls(
            status=ToolStatus.FAILED,
            tool_name=tool_name,
            error=error,
            user_facing_warning=user_facing_warning or "该服务暂时不可用，回答可能不完整。",
            confidence=Confidence.NONE,
            latency_ms=latency_ms,
        )

    @classmethod
    def degraded(cls, tool_name: str, data: Any, warning: str,
                 sources: Optional[List[SourceTrace]] = None,
                 confidence: Confidence = Confidence.LOW,
                 user_facing_warning: Optional[str] = None,
                 latency_ms: float = 0.0) -> MedToolResult:
        return cls(
            status=ToolStatus.DEGRADED,
            tool_name=tool_name,
            data=data,
            warning=warning,
            user_facing_warning=user_facing_warning or "部分检索源暂时不可用，结果仅供参考。",
            sources=sources or [],
            confidence=confidence,
            latency_ms=latency_ms,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "tool_name": self.tool_name,
            "confidence": self.confidence.value if self.confidence else "none",
            "sources": [s.to_dict() for s in self.sources],
            "user_facing_warning": self.user_facing_warning,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
        }

    def __bool__(self) -> bool:
        return self.status in (ToolStatus.SUCCESS, ToolStatus.DEGRADED)


# =========================================================================
# 医疗场景 Harness 配置
# =========================================================================

_DEFAULT_TOOL_TIMEOUTS: Dict[str, int] = {
    "kg_search": 15000,         # Neo4j 知识图谱查询
    "qa_search": 5000,          # Milvus + ES 双路检索
    "es_search": 5000,          # ES BM25
    "llm_generate": 60000,      # LLM 生成（含流式）
    "llm_route": 10000,         # LLM 路由决策
    "dosage_calculator": 2000,  # 本地剂量计算
    "normal_range": 2000,       # 本地正常值查询
    "department_guide": 2000,   # 本地科室导诊
}

_DEFAULT_RETRY_POLICIES: Dict[str, dict] = {
    "retrieval": {"max_retries": 2, "delay_ms": 500, "backoff": 2.0},
    "llm": {"max_retries": 1, "delay_ms": 1000, "backoff": 2.0},
    "local_tool": {"max_retries": 0, "delay_ms": 0, "backoff": 1.0},
}

_DEFAULT_RETRIEVAL_FALLBACK: List[str] = ["kg", "qa", "es", "cached"]
_DEFAULT_LLM_FALLBACK: List[str] = ["primary", "ollama", "cached"]


@dataclass
class HarnessConfig:
    """医疗场景的容错执行配置。

    预置了医疗各组件的典型超时和重试策略，可直接使用。
    区别于 AGI-saber：多了 tool_timeouts 和 retry_policies 的细粒度控制。
    """
    tool_timeouts: Dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_TOOL_TIMEOUTS))
    retry_policies: Dict[str, dict] = field(default_factory=lambda: dict(_DEFAULT_RETRY_POLICIES))
    retrieval_fallback: List[str] = field(default_factory=lambda: list(_DEFAULT_RETRIEVAL_FALLBACK))
    llm_fallback: List[str] = field(default_factory=lambda: list(_DEFAULT_LLM_FALLBACK))
    safety_detection_required: bool = True
    max_parallel: int = 4
    enable_racing: bool = True
    snapshot_enabled: bool = True

    # 允许在构造时覆盖指定超时
    override_timeouts: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.override_timeouts:
            self.tool_timeouts.update(self.override_timeouts)

    def get_timeout(self, tool_name: str, default: int = 30000) -> int:
        return self.tool_timeouts.get(tool_name, default)

    def get_retry_policy(self, category: str = "retrieval") -> dict:
        return self.retry_policies.get(category, self.retry_policies["retrieval"])


# =========================================================================
# 医疗问答流程状态机
# =========================================================================


class MedPhase(str, enum.Enum):
    """医疗问答的执行阶段（ReAct 统一架构）。

    所有查询统一进入 ReAct 循环，LLM 自主决定调工具还是直接回答。
      risk_detect(输入安全检测) → route(仅元数据，不选模式)
      → react_loop(ReAct 推理循环) → safety_check(输出安全检测) → complete

    RETRIEVE/RERANK/ASSEMBLE/GENERATE 已由 REACT_LOOP 统一承载，
    RAG 检索作为 retrieve_knowledge 工具在循环内调用。
    """
    IDLE = "idle"
    RISK_DETECT = "risk_detect"    # 输入安全检测（检测紧急/风险关键词）
    ROUTE = "route"                 # 路由元数据（query_type、数据源选择，不选执行模式）
    REACT_LOOP = "react_loop"       # ReAct 推理循环（LLM 自主决策调工具或回答）
    SAFETY_CHECK = "safety_check"   # 输出安全检测（追加免责声明）
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class MedStateMachine:
    """医疗问答流程状态机（ReAct 统一架构）。

    强制流程约束：
      1. REACT_LOOP 必须在 RISK_DETECT 之后执行（输入安全检测前置）
      2. SAFETY_CHECK 是 COMPLETE 的前置条件（医疗场景必须检查输出安全）
      3. 可以通过 ``force_transition`` 跳过中间阶段（用于恢复/测试）
    """
    session_id: str
    current_phase: MedPhase = MedPhase.IDLE
    history: List[dict] = field(default_factory=list)
    risk_level: Optional[str] = None
    risk_keywords: List[str] = field(default_factory=list)
    retry_history: List[dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _risk_detected: bool = False
    _safety_checked: bool = False

    def transition(self, target: MedPhase) -> None:
        """阶段转换，强制执行医疗场景的约束。"""
        now = time.time()

        if target == MedPhase.REACT_LOOP and not self._risk_detected:
            raise ValueError(
                "RISK_DETECT must be completed before REACT_LOOP (medical safety requirement)"
            )

        if target == MedPhase.COMPLETE and not self._safety_checked:
            raise ValueError(
                "SAFETY_CHECK must be completed before COMPLETE (medical safety requirement)"
            )

        # 进入 RISK_DETECT 或 SAFETY_CHECK 阶段即标记对应检查已完成
        if target == MedPhase.RISK_DETECT:
            self._risk_detected = True
        if target == MedPhase.SAFETY_CHECK:
            self._safety_checked = True

        self.current_phase = target
        self.history.append({"phase": target.value, "timestamp": now})

        if target == MedPhase.COMPLETE and len(self.history) > 1:
            self.metadata["total_duration_ms"] = (now - self.history[0]["timestamp"]) * 1000

    def force_transition(self, target: MedPhase) -> None:
        """跳过约束强制转换（仅用于测试/恢复）。"""
        now = time.time()
        self.current_phase = target
        self.history.append({"phase": target.value, "timestamp": now, "forced": True})

    def record_risk(self, level: str, keywords: List[str]) -> None:
        """记录安全检测结果。"""
        self.risk_level = level
        self.risk_keywords = keywords
        self._risk_detected = True
        self.metadata["risk_detected_at"] = time.time()

    def record_safety_checked(self) -> None:
        """标记安全检测已完成。"""
        self._safety_checked = True
        self.metadata["safety_checked_at"] = time.time()

    def record_retry(self, tool: str, attempt: int, reason: str) -> None:
        """记录重试事件。"""
        self.retry_history.append({
            "tool": tool,
            "attempt": attempt,
            "reason": reason,
            "timestamp": time.time(),
        })

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "current_phase": self.current_phase.value,
            "risk_level": self.risk_level,
            "risk_keywords": self.risk_keywords,
            "retry_count": len(self.retry_history),
            "retry_history": self.retry_history,
            "total_duration_ms": self.metadata.get("total_duration_ms"),
            "history": self.history,
        }
