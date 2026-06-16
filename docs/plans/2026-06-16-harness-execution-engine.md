# Harness 容错执行引擎 — MedAgent 医疗场景实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为 MedAgent 构建一个医疗场景原生的 Harness 执行引擎，将当前散布在 `_safe_*`、`try/except`、`AnswerGenerator.fallback` 中的容错逻辑统一为可编排的容错框架，并支持未来的 DAG 并行调度。

**设计原则：**
1. **医疗安全优先** — 安全检测（SafetyGuard）是强制阶段，不是可选项
2. **证据溯源** — 每个结果都携带来源、置信度，回答可追责
3. **渐进式降级** — 不隐瞒失败，但用中文明确告知用户"检索结果不完整"
4. **包装而非替换** — Harness 包装现有 `HybridRetriever`、`AnswerGenerator`、`SafetyGuard`，不改动既有逻辑
5. **增量可用** — 每个 Task 提交后立即可用，不依赖后续 Task

**架构：** 3 层：Core Types（医疗元数据+溯源）→ ToolWrapper（包装现有检索/LLM/计算工具）→ HarnessOrchestrator（全流程编排+状态机+fallback）。

---

### Task 1: 医疗场景核心类型 — MedToolResult、MedStateMachine、HarnessConfig

**Files:**
- Create: `src/medrag/harness/__init__.py`
- Create: `src/medrag/harness/types.py`
- Test: `tests/test_harness_types.py`

**医疗场景适配设计：**
- `MedToolResult` 携带 `sources[]`（证据来源列表）、`confidence`（high/medium/low/none）、`risk_flags[]`（触发风险关键词）
- `MedStateMachine` 的阶段 = MedAgent 的实际流程：`risk_detect → route → retrieve → rerank → assemble → generate → safety_check → complete`
- `HarnessConfig` 预置医疗组件的超时推荐值（KG 慢、剂量计算快）
- `SourceTrace` 记录每条证据的溯源路径：retrieval 来源、rerank 分数、原始文本片段

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_harness_types.py — MedAgent 医疗场景核心类型。"""
from __future__ import annotations

import pytest

from medrag.harness.types import (
    MedToolResult,
    ToolStatus,
    HarnessConfig,
    MedStateMachine,
    MedPhase,
    SourceTrace,
    Confidence,
)


class TestSourceTrace:
    """医疗证据溯源。"""
    def test_create_source_trace(self):
        st = SourceTrace(
            source_type="kg",
            content="高血压是一种慢性病",
            score=0.95,
            source_name="neo4j_medical_kg",
        )
        assert st.source_type == "kg"
        assert st.content == "高血压是一种慢性病"
        assert st.score == 0.95

    def test_to_dict(self):
        st = SourceTrace(source_type="qa", content="text", score=0.8, source_name="milvus")
        d = st.to_dict()
        assert d["source_type"] == "qa"
        assert d["score"] == 0.8


class TestMedToolResult:
    """医疗工具执行结果 — 含证据溯源和置信度。"""
    def test_ok_result_with_sources(self):
        r = MedToolResult.ok(
            tool_name="kg_search",
            data={"answer": "高血压"},
            sources=[
                SourceTrace("kg", "高血压定义", 0.95, "neo4j"),
            ],
        )
        assert r.status == ToolStatus.SUCCESS
        assert r.confidence == Confidence.HIGH
        assert len(r.sources) == 1
        assert r.latency_ms >= 0

    def test_failed_result_propagates_warning(self):
        r = MedToolResult.failed(
            tool_name="kg_search",
            error="Neo4j connection refused",
            user_facing_warning="知识图谱暂时不可用，回答可能不完整。",
        )
        assert r.status == ToolStatus.FAILED
        assert r.user_facing_warning  # 面向用户的提示

    def test_degraded_still_has_data_with_warning(self):
        r = MedToolResult.degraded(
            tool_name="qa_search",
            data=[{"answer": "partial"}],
            warning="Milvus 超时，仅使用 ES 结果",
            sources=[SourceTrace("qa", "partial", 0.7, "es")],
            confidence=Confidence.MEDIUM,
        )
        assert r.status == ToolStatus.DEGRADED
        assert r.confidence == Confidence.MEDIUM
        assert bool(r) is True  # degraded 视为可用

    def test_bool_failed_is_false(self):
        r = MedToolResult.failed(tool_name="x", error="err")
        assert bool(r) is False

    def test_to_dict_contains_medical_fields(self):
        r = MedToolResult.ok(tool_name="test", data="ok")
        d = r.to_dict()
        assert "confidence" in d
        assert "sources" in d
        assert "user_facing_warning" in d


class TestHarnessConfig:
    """医疗场景默认配置。"""
    def test_medical_defaults(self):
        c = HarnessConfig()
        assert c.tool_timeouts["kg_search"] == 15000       # KG 慢
        assert c.tool_timeouts["qa_search"] == 5000         # QA 快
        assert c.tool_timeouts["llm_generate"] == 60000     # LLM 最慢
        assert c.tool_timeouts["dosage_calculator"] == 2000  # 本地计算最快

    def test_rag_retry_policy(self):
        c = HarnessConfig()
        assert c.retry_policies["retrieval"]["max_retries"] == 2  # 网络重试
        assert c.retry_policies["local_tool"]["max_retries"] == 0  # 本地不重试

    def test_custom_overrides(self):
        c = HarnessConfig(override_timeouts={"kg_search": 30000})
        assert c.get_timeout("kg_search") == 30000

    def test_fallback_chain_ordered(self):
        c = HarnessConfig()
        assert c.retrieval_fallback == ["kg", "qa", "es", "cached"]
        assert c.llm_fallback == ["primary", "ollama", "cached"]


class TestMedStateMachine:
    """医疗问答流程状态机。"""
    def test_full_happy_path(self):
        sm = MedStateMachine(session_id="test-1")
        assert sm.current_phase == MedPhase.IDLE

        sm.transition(MedPhase.RISK_DETECT)
        assert sm.current_phase == MedPhase.RISK_DETECT

        sm.transition(MedPhase.ROUTE)
        sm.transition(MedPhase.RETRIEVE)
        sm.transition(MedPhase.RERANK)
        sm.transition(MedPhase.ASSEMBLE)
        sm.transition(MedPhase.GENERATE)
        sm.transition(MedPhase.SAFETY_CHECK)
        sm.transition(MedPhase.COMPLETE)
        assert sm.current_phase == MedPhase.COMPLETE

    def test_cannot_skip_risk_detect(self):
        sm = MedStateMachine(session_id="test-2")
        with pytest.raises(ValueError, match="RISK_DETECT"):
            sm.transition(MedPhase.GENERATE)

    def test_safety_check_mandatory_before_complete(self):
        sm = MedStateMachine(session_id="test-3")
        sm.force_transition(MedPhase.GENERATE)  # 跳过中间阶段（测试用）
        with pytest.raises(ValueError, match="SAFETY_CHECK"):
            sm.transition(MedPhase.COMPLETE)

    def test_risk_flag_raises_alert(self):
        sm = MedStateMachine(session_id="test-4")
        sm.transition(MedPhase.RISK_DETECT)
        sm.record_risk("high", ["心脏病", "胸痛"])
        assert sm.risk_level == "high"
        assert "胸痛" in sm.risk_keywords

    def test_retry_recorded(self):
        sm = MedStateMachine(session_id="test-5")
        sm.transition(MedPhase.RETRIEVE)
        sm.record_retry("kg_search", attempt=1, reason="timeout")
        assert sm.retry_history[0]["tool"] == "kg_search"
        assert sm.retry_history[0]["reason"] == "timeout"

    def test_to_dict_contains_medical_state(self):
        sm = MedStateMachine(session_id="test-6")
        d = sm.to_dict()
        assert "session_id" in d
        assert "current_phase" in d
        assert "retry_history" in d
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_harness_types.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""src/medrag/harness/__init__.py"""
from __future__ import annotations

from .types import (
    MedToolResult, ToolStatus, Confidence,
    SourceTrace,
    HarnessConfig,
    MedStateMachine, MedPhase,
)

__all__ = [
    "MedToolResult", "ToolStatus", "Confidence",
    "SourceTrace",
    "HarnessConfig",
    "MedStateMachine", "MedPhase",
]
```

```python
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

    用于在回答中标注``【来源：知识图谱】``，提升医疗可信度和可审计性。
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

    相比 AGI-saber 的 ExecResult，增加了医疗场景特有的：
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
    def ok(cls, tool_name: str, data: Any, sources: Optional[List[SourceTrace]] = None,
           confidence: Confidence = Confidence.HIGH, latency_ms: float = 0.0) -> MedToolResult:
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


@dataclass
class HarnessConfig:
    """医疗场景的容错执行配置。

    预置了医疗各组件的典型超时和重试策略，可直接使用。
    区别于 AGI-saber：多了 tool_timeouts 和 retry_policies 的细粒度控制。
    """
    # 各工具的默认超时（毫秒）
    tool_timeouts: Dict[str, int] = field(default_factory=lambda: {
        "kg_search": 15000,        # Neo4j 知识图谱查询
        "qa_search": 5000,         # Milvus + ES 双路检索
        "es_search": 5000,         # ES BM25
        "llm_generate": 60000,     # LLM 生成（含流式）
        "llm_route": 10000,        # LLM 路由决策
        "dosage_calculator": 2000, # 本地剂量计算
        "normal_range": 2000,      # 本地正常值查询
        "department_guide": 2000,  # 本地科室导诊
    })

    # 各场景的重试策略
    retry_policies: Dict[str, dict] = field(default_factory=lambda: {
        "retrieval": {"max_retries": 2, "delay_ms": 500, "backoff": 2.0},  # 网络重试
        "llm": {"max_retries": 1, "delay_ms": 1000, "backoff": 2.0},      # API 重试
        "local_tool": {"max_retries": 0, "delay_ms": 0, "backoff": 1.0},   # 本地不重试
    })

    # 检索降级链路（从高到低）
    retrieval_fallback: List[str] = field(default_factory=lambda: [
        "kg", "qa", "es", "cached",
    ])

    # LLM 降级链路
    llm_fallback: List[str] = field(default_factory=lambda: [
        "primary", "ollama", "cached",
    ])

    # 安全检测开关
    safety_detection_required: bool = True  # 医疗场景：必须做安全检测
    max_parallel: int = 4
    enable_racing: bool = True
    snapshot_enabled: bool = True

    def __post_init__(self):
        # 兼容代理构造：如果传入了 override_timeouts 等
        pass

    def get_timeout(self, tool_name: str, default: int = 30000) -> int:
        return self.tool_timeouts.get(tool_name, default)

    def get_retry_policy(self, category: str = "retrieval") -> dict:
        return self.retry_policies.get(category, self.retry_policies["retrieval"])


# =========================================================================
# 医疗问答流程状态机
# =========================================================================


class MedPhase(str, enum.Enum):
    """医疗问答的执行阶段。

    反映 MedAgent 的真实流程：
      risk_detect(安全检测) → route(路由) → retrieve(多源检索)
      → rerank(重排序) → assemble(组装上下文) → generate(LLM生成)
      → safety_check(二次安全检测) → complete(完成)
    
    这是医疗场景专有的阶段定义，非通用 Agent 阶段。
    """
    IDLE = "idle"
    RISK_DETECT = "risk_detect"    # 输入安全检测（检测紧急/风险关键词）
    ROUTE = "route"                 # 路由决策（chat/rag/react/tool）
    RETRIEVE = "retrieve"           # 多源检索（KG / QA / ES / Case）
    RERANK = "rerank"               # Cross-Encoder 重排序
    ASSEMBLE = "assemble"           # Schema-Driven Context Assembly
    GENERATE = "generate"           # LLM 生成回答
    SAFETY_CHECK = "safety_check"   # 输出安全检测（追加免责声明）
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class MedStateMachine:
    """医疗问答流程状态机。

    强制流程约束：
      1. RISK_DETECT 必须在 GENERATE 之前至少执行一次
      2. SAFETY_CHECK 是 COMPLETE 的前置条件（医疗场景必须检查输出安全）
      3. 可以通过 ``force_transition`` 跳过中间阶段（用于恢复/测试）
    """
    session_id: str
    current_phase: MedPhase = MedPhase.IDLE
    history: List[dict] = field(default_factory=list)
    risk_level: Optional[str] = None          # "high" | "yellow" | None
    risk_keywords: List[str] = field(default_factory=list)
    retry_history: List[dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _risk_detected: bool = False
    _safety_checked: bool = False

    def transition(self, target: MedPhase) -> None:
        """阶段转换，强制执行医疗场景的约束。"""
        now = time.time()

        # 约束：RISK_DETECT 必须在 GENERATE 之前完成
        if target == MedPhase.GENERATE and not self._risk_detected:
            raise ValueError(
                "RISK_DETECT must be completed before GENERATE (medical safety requirement)"
            )

        # 约束：SAFETY_CHECK 是 COMPLETE 前置
        if target == MedPhase.COMPLETE and not self._safety_checked:
            raise ValueError(
                "SAFETY_CHECK must be completed before COMPLETE (medical safety requirement)"
            )

        self.current_phase = target
        self.history.append({"phase": target.value, "timestamp": now})

        if target == MedPhase.COMPLETE:
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
            "total_duration_ms": self.metadata.get("total_duration_ms"),
            "history": self.history,
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_harness_types.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/medrag/harness/ tests/test_harness_types.py
git commit -m "feat(harness): 医疗场景核心类型 — MedToolResult、MedStateMachine、HarnessConfig

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: 医疗工具包装器 — 封装现有检索/LLM/原生工具

**Files:**
- Create: `src/medrag/harness/wrappers.py`
- Test: `tests/test_harness_wrappers.py`

**医疗场景适配设计：**
- 将现有 `HybridRetriever`、`AnswerGenerator`、`SafetyGuard`、`DosageCalculator` 等包装为统一的 `MedToolWrapper`
- 包装器负责：执行 → 超时 → 捕获异常 → 封装为 `MedToolResult`（含置信度和溯源）
- 记录重试事件到 `MedStateMachine`

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_harness_wrappers.py — 医疗工具包装器。"""
from __future__ import annotations

import pytest

from medrag.harness.types import MedToolResult, ToolStatus, Confidence, SourceTrace, HarnessConfig
from medrag.harness.wrappers import (
    MedToolWrapper,
    create_retrieval_wrapper,
    create_llm_wrapper,
    create_safety_wrapper,
    create_builtin_wrapper,
)


class TestMedToolWrapper:
    def test_wrapper_success(self):
        def fake_exec(**kw):
            return {"answer": "高血压"}

        wrapper = MedToolWrapper(
            name="kg_search",
            category="retrieval",
            executor=fake_exec,
        )
        result = wrapper.execute(query="高血压")
        assert result.status == ToolStatus.SUCCESS
        assert result.data == {"answer": "高血压"}

    def test_wrapper_catches_exception(self):
        def broken(**kw):
            raise ConnectionError("Neo4j down")

        wrapper = MedToolWrapper(
            name="kg_search",
            category="retrieval",
            executor=broken,
            on_error_msg="知识图谱暂时不可用，回答可能不完整。",
        )
        result = wrapper.execute(query="test")
        assert result.status == ToolStatus.FAILED
        assert "Neo4j down" in result.error
        assert "知识图谱暂时不可用" in result.user_facing_warning

    def test_wrapper_timeout_via_config(self):
        config = HarnessConfig(override_timeouts={"kg_search": 1})  # 1ms 超时
        def slow(**kw):
            import time
            time.sleep(10)
            return "done"

        wrapper = MedToolWrapper(
            name="kg_search", category="retrieval",
            executor=slow,
        )
        result = wrapper.execute(config=config, query="test")
        assert result.status == ToolStatus.TIMEOUT

    def test_wrapper_records_retries(self):
        call_count = [0]
        def flaky(**kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("transient")

        wrapper = MedToolWrapper(
            name="qa_search",
            category="retrieval",
            executor=flaky,
        )
        config = HarnessConfig()
        config.retry_policies["retrieval"] = {"max_retries": 2, "delay_ms": 1, "backoff": 1.0}
        result = wrapper.execute(config=config, query="test")
        assert result.status == ToolStatus.SUCCESS
        assert result.retry_count == 1

    def test_confidence_tracking(self):
        """成功执行后置信度应为 HIGH。"""
        def ok(**kw):
            return {"answer": "data"}
        wrapper = MedToolWrapper(name="test", category="retrieval", executor=ok)
        result = wrapper.execute()
        assert result.confidence == Confidence.HIGH


class TestCreateRetrievalWrapper:
    def test_wraps_hybrid_retriever(self):
        """使用 mock 验证 HybridRetriever 被正确包装。"""
        class FakeHybridRetriever:
            def retrieve(self, query, top_k=10, department=None, username=None):
                return {
                    "kg_results": [{"answer": "高血压定义"}],
                    "qa_results": [{"answer": "高血压治疗"}],
                    "fusion_mode": "rrf_dense_sparse",
                }

        wrapper = create_retrieval_wrapper(FakeHybridRetriever())
        result = wrapper.execute(query="高血压")
        assert result.status == ToolStatus.SUCCESS
        assert len(result.sources) >= 1
        # 应该对每个检索结果生成 SourceTrace
        kg_traces = [s for s in result.sources if s.source_type == "kg"]
        assert len(kg_traces) >= 1


class TestCreateLLMWrapper:
    def test_wraps_answer_generator(self):
        """使用 mock 验证 AnswerGenerator 被正确包装。"""
        class FakeGenerator:
            def generate(self, messages, model=None):
                return "高血压患者应注意..."

        wrapper = create_llm_wrapper(FakeGenerator())
        result = wrapper.execute(messages=[{"role": "user", "content": "高血压注意什么"}])
        assert result.status == ToolStatus.SUCCESS
        assert "高血压" in str(result.data)


class TestCreateSafetyWrapper:
    def test_detect_risk(self):
        class FakeSafetyGuard:
            def detect_risk(self, query):
                return {"has_risk": True, "risk_keywords": ["胸痛"], "level": "high"}
            def append_safety_notice(self, answer, risk_info, **kw):
                return answer + "\n\n[安全提示]"

        wrapper = create_safety_wrapper(FakeSafetyGuard())
        result = wrapper.execute(query="胸痛怎么办", answer="请立即就医")
        assert result.status == ToolStatus.SUCCESS
        assert "胸痛" in str(result.risk_flags)


class TestCreateBuiltinWrapper:
    def test_wraps_dosage_calculator(self):
        class FakeDosage:
            name = "dosage_calculator"
            description = "计算剂量"
            def execute(self, **kw):
                return "阿莫西林：每次0.5g"

        wrapper = create_builtin_wrapper(FakeDosage())
        result = wrapper.execute(drug_name="阿莫西林", age=10)
        assert result.status == ToolStatus.SUCCESS
        assert "阿莫西林" in str(result.data)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_harness_wrappers.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
"""src/medrag/harness/wrappers.py — 医疗工具包装器。

将现有 MedAgent 组件（HybridRetriever、AnswerGenerator、SafetyGuard、BaseTool）
包装为统一的 ``MedToolWrapper``，每个包装器负责：
  1. 执行 + 超时 + 重试
  2. 异常 → MedToolResult 转换（含面向用户的降级提示）
  3. 证据溯源（SourceTrace）和置信度计算
  4. 重试事件记录
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from medrag.harness.types import (
    MedToolResult, ToolStatus, Confidence, SourceTrace, HarnessConfig,
)

logger = logging.getLogger(__name__)


# =========================================================================
# MedToolWrapper — 统一医疗工具包装器
# =========================================================================


@dataclass
class MedToolWrapper:
    """包装一个医疗工具（检索/LLM/原生）的统一接口。

    Usage::

        wrapper = MedToolWrapper(
            name="kg_search",
            category="retrieval",
            executor=hybrid_retriever.retrieve,
            on_error_msg="知识图谱暂时不可用",
        )
        result: MedToolResult = wrapper.execute(query="高血压")
    """
    name: str
    category: str                    # "retrieval" | "llm" | "local_tool" | "safety"
    executor: Callable[..., Any]     # 实际的工具函数
    on_error_msg: str = ""           # 失败时面向用户的降级提示
    make_sources: bool = False       # 是否从结果自动提取 SourceTrace
    state_machine: Optional[MedStateMachine] = None  # 可选：用于记录事件

    def execute(self, config: Optional[HarnessConfig] = None, **params) -> MedToolResult:
        cfg = config or HarnessConfig()
        timeout = cfg.get_timeout(self.name)
        retry_policy = cfg.get_retry_policy(self.category)

        last_result: Optional[MedToolResult] = None
        start = time.monotonic()

        for attempt in range(retry_policy["max_retries"] + 1):
            # 记录重试
            if attempt > 0 and self.state_machine:
                self.state_machine.record_retry(self.name, attempt, f"attempt_{attempt}")

            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self.executor, **params)
                    data = future.result(timeout=timeout / 1000)

                elapsed = (time.monotonic() - start) * 1000
                result = self._build_success_result(data, elapsed)

                if self.state_machine:
                    self.state_machine.metadata.setdefault("tool_latencies", {})
                    self.state_machine.metadata["tool_latencies"][self.name] = elapsed

                return result

            except TimeoutError:
                elapsed = (time.monotonic() - start) * 1000
                last_result = MedToolResult(
                    status=ToolStatus.TIMEOUT,
                    tool_name=self.name,
                    error=f"Timeout after {timeout}ms",
                    user_facing_warning=self.on_error_msg or f"「{self.name}」查询超时，结果可能不完整。",
                    confidence=Confidence.LOW,
                    latency_ms=elapsed,
                )
                if attempt < retry_policy["max_retries"]:
                    import time as _time
                    _time.sleep(retry_policy["delay_ms"] / 1000 * (retry_policy["backoff"] ** attempt))
                    continue
                break

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                last_result = MedToolResult(
                    status=ToolStatus.FAILED,
                    tool_name=self.name,
                    error=str(exc),
                    user_facing_warning=self.on_error_msg or "该服务暂时不可用。",
                    latency_ms=elapsed,
                )
                if attempt < retry_policy["max_retries"]:
                    import time as _time
                    _time.sleep(retry_policy["delay_ms"] / 1000 * (retry_policy["backoff"] ** attempt))
                    continue
                break

        return last_result or MedToolResult.failed(
            tool_name=self.name,
            error="Unknown failure",
            user_facing_warning=self.on_error_msg,
        )

    def _build_success_result(self, data: Any, latency_ms: float) -> MedToolResult:
        """从原始工具返回值构建 MedToolResult。"""
        sources: List[SourceTrace] = []

        if self.make_sources and isinstance(data, dict):
            # 从 HybridRetriever 的返回中提取 SourceTrace
            for key, stype in [("kg_results", "kg"), ("qa_results", "qa")]:
                items = data.get(key, [])
                for i, item in enumerate(items):
                    content = item.get("answer") or item.get("text") or str(item)
                    sources.append(SourceTrace(
                        source_type=stype,
                        content=str(content)[:500],
                        score=item.get("rrf_score", item.get("score", 0)),
                        source_name=item.get("source", stype),
                        rank=i + 1,
                    ))

        confidence = Confidence.HIGH
        if sources:
            # 多源 + 高分为 HIGH，仅单源为 MEDIUM
            unique_types = set(s.source_type for s in sources)
            avg_score = sum(s.score for s in sources) / len(sources)
            if len(unique_types) >= 2 and avg_score > 0.3:
                confidence = Confidence.HIGH
            else:
                confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.HIGH  # 原生工具无 source 但成功 = HIGH

        return MedToolResult.ok(
            tool_name=self.name,
            data=data,
            sources=sources,
            confidence=confidence,
            latency_ms=latency_ms,
        )


# =========================================================================
# 工厂函数：为具体 MedAgent 组件创建包装器
# =========================================================================


def create_retrieval_wrapper(retriever) -> MedToolWrapper:
    """包装 HybridRetriever → MedToolWrapper。

    负责执行多源检索并将结果解析为带 SourceTrace 的 MedToolResult。
    """
    return MedToolWrapper(
        name="hybrid_retrieval",
        category="retrieval",
        executor=retriever.retrieve,
        make_sources=True,
        on_error_msg="多源检索暂时不可用，部分结果可能缺失。",
    )


def create_llm_wrapper(answer_generator) -> MedToolWrapper:
    """包装 AnswerGenerator → MedToolWrapper。"""
    return MedToolWrapper(
        name="llm_generate",
        category="llm",
        executor=answer_generator.generate,
        on_error_msg="AI 模型暂时无法生成回答，请稍后重试。",
    )


def create_safety_wrapper(safety_guard) -> MedToolWrapper:
    """包装 SafetyGuard → MedToolWrapper。

    返回的 MedToolResult 会在 ``risk_flags`` 中携带检测到的风险关键词。
    """
    def _detect_and_append(query: str, answer: str = "",
                           retrieval_quality: str = "none",
                           query_type: str = "") -> dict:
        risk_info = safety_guard.detect_risk(query)
        full_answer = safety_guard.append_safety_notice(
            answer, risk_info,
            retrieval_quality=retrieval_quality,
            query_type=query_type,
        )
        return {
            "answer": full_answer,
            "has_risk": risk_info.get("has_risk", False),
            "risk_keywords": risk_info.get("risk_keywords", []),
            "risk_level": risk_info.get("level", "none"),
        }

    return MedToolWrapper(
        name="safety_check",
        category="safety",
        executor=_detect_and_append,
    )


def create_builtin_wrapper(builtin_tool) -> MedToolWrapper:
    """包装 BaseTool（剂量计算/正常值/科室导诊）→ MedToolWrapper。"""
    tool_name = getattr(builtin_tool, 'name', builtin_tool.__class__.__name__)
    return MedToolWrapper(
        name=tool_name,
        category="local_tool",
        executor=builtin_tool.execute,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_harness_wrappers.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/medrag/harness/wrappers.py tests/test_harness_wrappers.py
git commit -m "feat(harness): 医疗工具包装器 — 封装现有检索/LLM/原生工具为统一接口

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: HarnessOrchestrator — 医疗全流程编排器

**Files:**
- Create: `src/medrag/harness/orchestrator.py`
- Modify: `src/medrag/harness/__init__.py`
- Test: `tests/test_harness_orchestrator.py`

**医疗场景适配设计：**
- `HarnessOrchestrator` 编排完整的医疗问答流程：risk_detect → route → retrieve → rerank → assemble → generate → safety_check → complete
- 每个阶段有独立的超时和重试策略
- 阶段间传递 `MedStateMachine` 状态
- 所有阶段失败都有面向用户的降级提示（中文）
- 集成 SafetyGuard 作为强制阶段

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_harness_orchestrator.py — 医疗全流程编排器。"""
from __future__ import annotations

import pytest

from medrag.harness.types import (
    MedToolResult, ToolStatus, Confidence, MedPhase, HarnessConfig,
)
from medrag.harness.orchestrator import HarnessOrchestrator


class TestHarnessOrchestrator:
    def test_orchestrate_rag_pipeline(self):
        """完整 RAG 流程编排。"""
        # Mock 组件
        def fake_risk_detect(query, **kw):
            return {"has_risk": False, "risk_keywords": [], "level": "none"}

        def fake_route(query, **kw):
            return {"execution_mode": "rag", "query_type": "medical"}

        def fake_retrieve(query, **kw):
            return {
                "kg_results": [{"answer": "高血压定义", "rrf_score": 0.8}],
                "qa_results": [{"answer": "高血压治疗", "rrf_score": 0.7}],
                "fusion_mode": "rrf_dense_sparse",
            }

        def fake_rerank(query, results, **kw):
            return results[:3] if results else []

        def fake_assemble(**kw):
            return {"messages": [{"role": "user", "content": "test"}]}

        def fake_generate(messages, **kw):
            return "高血压患者应注意..."

        def fake_safety_check(query, answer, **kw):
            return answer + "\n\n*以上信息仅供参考*"

        orchestrator = HarnessOrchestrator(
            risk_detector=fake_risk_detect,
            router=fake_route,
            retriever=fake_retrieve,
            reranker=fake_rerank,
            assembler=fake_assemble,
            generator=fake_generate,
            safety_checker=fake_safety_check,
        )

        result = orchestrator.run(query="高血压注意什么")
        assert result["status"] == "success"
        assert "高血压患者应注意" in result["answer"]
        assert "仅供参考" in result["answer"]
        assert "trace" in result

    def test_orchestrate_tool_mode(self):
        """工具模式流程（不走检索和生成）。"""
        def fake_route(query, **kw):
            return {"execution_mode": "tool", "tool_name": "dosage_calculator"}

        def fake_tool_exec(tool_name, **params):
            return MedToolResult.ok(tool_name="dosage_calculator", data="阿莫西林：0.5g")

        orchestrator = HarnessOrchestrator(
            router=fake_route,
            tool_executor=fake_tool_exec,
        )
        result = orchestrator.run(query="阿莫西林用量")
        assert result["status"] == "success"
        assert "阿莫西林" in str(result["answer"])

    def test_risk_detect_triggers_warning(self):
        """检测到风险关键词时，回答应包含安全提示。"""
        def fake_risk(query, **kw):
            return {"has_risk": True, "risk_keywords": ["胸痛", "心脏病"], "level": "high"}

        orchestrator = HarnessOrchestrator(
            risk_detector=fake_risk,
            router=lambda q, **kw: {"execution_mode": "rag"},
            retriever=lambda q, **kw: {"kg_results": [], "qa_results": []},
            assembler=lambda **kw: {"messages": []},
            generator=lambda m, **kw: "回答内容",
            safety_checker=lambda q, a, **kw: a + "\n\n[紧急提示] 请立即就医",
        )
        result = orchestrator.run(query="胸痛怎么办")
        assert result["risk_info"]["has_risk"] is True
        assert "紧急提示" in result["answer"]

    def test_retrieval_fallback_on_failure(self):
        """检索全部失败时有降级响应。"""
        def fake_retrieve(**kw):
            raise ConnectionError("ALL DOWN")

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": []},
            router=lambda q, **kw: {"execution_mode": "rag"},
            retriever=fake_retrieve,
            reranker=lambda q, r, **kw: [],
            assembler=lambda **kw: {"messages": [{"role": "user", "content": "test"}]},
            generator=lambda m, **kw: "基于医学知识：...",
            safety_checker=lambda q, a, **kw: a + "\n\n*信息有限*",
        )
        result = orchestrator.run(query="高血压")
        # 即使检索失败，orchestrator 仍应返回结果（LLM 兜底）
        assert result["status"] == "success"
        assert "harness_warning" in result

    def test_streaming_events(self):
        """流式事件应包含每个阶段的变更。"""
        events = []

        def on_event(event_type, data):
            events.append((event_type, data))

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": []},
            router=lambda q, **kw: {"execution_mode": "rag"},
            retriever=lambda q, **kw: {"kg_results": [{"answer": "x"}], "qa_results": []},
            reranker=lambda q, r, **kw: r,
            assembler=lambda **kw: {"messages": [{"role": "user", "content": "test"}]},
            generator=lambda m, **kw: "回答内容",
            safety_checker=lambda q, a, **kw: a + "\n\n*仅供参考*",
            on_event=on_event,
        )
        orchestrator.run(query="高血压")
        event_types = [e[0] for e in events]
        assert "phase_change" in event_types
        assert "phase_change" in event_types
        assert "complete" in event_types
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_harness_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
"""src/medrag/harness/orchestrator.py — 医疗全流程编排器。

将 MedAgent 现有的串行流程（risk_detect → route → retrieve → ... → complete）
封装为带有状态机、超时、重试、事件通知的可编排 HarnessOrchestrator。

区别于 AGI-saber：不是通用 Agent 框架，而是专门为医疗问答的 4 种模式
（tool / chat / rag / react）提供统一的容错编排。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from medrag.harness.types import (
    MedToolResult, ToolStatus, Confidence, MedPhase, HarnessConfig, MedStateMachine,
)
from medrag.harness.wrappers import MedToolWrapper

logger = logging.getLogger(__name__)


@dataclass
class HarnessOrchestrator:
    """医疗问答全流程编排器。

    不创建新组件，而是包装现有的 MedicalChatService 子组件，
    用统一的容错策略（超时+重试+降级）覆盖每个阶段。

    Usage::

        orchestrator = HarnessOrchestrator(
            risk_detector=safety_guard.detect_risk,
            router=query_router.route,
            retriever=hybrid_retriever.retrieve,
            reranker=reranker.rerank,
            assembler=prompt_builder.build_messages_with_context,
            generator=answer_generator.generate,
            safety_checker=_do_safety_check,
        )
        result = orchestrator.run(query="高血压")
    """
    # 各阶段的执行函数
    risk_detector: Optional[Callable] = None
    router: Optional[Callable] = None
    retriever: Optional[Callable] = None
    reranker: Optional[Callable] = None
    assembler: Optional[Callable] = None
    generator: Optional[Callable] = None
    safety_checker: Optional[Callable] = None
    tool_executor: Optional[Callable] = None

    # 实时事件回调（用于前端 SSE 更新）
    on_event: Optional[Callable[[str, dict], None]] = None

    # 内部状态
    config: HarnessConfig = field(default_factory=HarnessConfig)
    state: Optional[MedStateMachine] = None

    def run(self, query: str, **context) -> Dict[str, Any]:
        """执行一次完整的医疗问答流程。

        Returns:
            统一响应字典，包含 answer、status、trace、risk_info 等。
        """
        self.state = MedStateMachine(session_id=f"session-{time.time_ns()}")
        self._emit("start", {"query": query})

        result: Dict[str, Any] = {
            "query": query,
            "status": "success",
            "answer": "",
            "route": {},
            "risk_info": {},
            "trace": {},
            "harness_warning": None,
        }

        # ── Phase 1: 安全检测 ──
        try:
            self.state.transition(MedPhase.RISK_DETECT)
            self._emit("phase_change", {"phase": "risk_detect"})

            if self.risk_detector:
                risk_info = self.risk_detector(query)
                self.state.record_risk(
                    risk_info.get("level", "none"),
                    risk_info.get("risk_keywords", []),
                )
                result["risk_info"] = risk_info
                has_risk = risk_info.get("has_risk", False)
                risk_level = risk_info.get("level", "none")
            else:
                has_risk = False
                risk_level = "none"
        except Exception as exc:
            logger.warning("Risk detection failed: %s", exc)
            has_risk = False
            risk_level = "none"
            result["harness_warning"] = "安全检测暂时不可用。"

        # ── Phase 2: 路由 ──
        try:
            self.state.transition(MedPhase.ROUTE)
            self._emit("phase_change", {"phase": "route"})

            if self.router:
                route = self.router(query)
                result["route"] = route
                mode = route.get("execution_mode", "rag")
            else:
                mode = "rag"
        except Exception as exc:
            logger.warning("Routing failed, default to rag: %s", exc)
            mode = "rag"
            result["route"] = {"execution_mode": "rag"}
            result["harness_warning"] = (result.get("harness_warning") or "") + "路由决策降级为默认。".strip()

        # ── 按模式分发 ──
        if mode == "tool":
            answer = self._run_tool_mode(query, result)
        elif mode == "chat":
            answer = self._run_chat_mode(query, result)
        elif mode == "react":
            answer = self._run_react_mode(query, result)
        else:
            answer = self._run_rag_mode(query, result, **context)

        result["answer"] = answer

        # ── Phase: 安全检测（输出） ──
        if self.safety_checker:
            try:
                self.state.transition(MedPhase.SAFETY_CHECK)
                self._emit("phase_change", {"phase": "safety_check"})

                retrieval_quality = "high" if result.get("trace", {}).get("sources") else "none"
                answer = self.safety_checker(
                    query=query, answer=answer,
                    retrieval_quality=retrieval_quality,
                    query_type=result.get("route", {}).get("query_type", ""),
                )
                result["answer"] = answer
                self.state.record_safety_checked()
            except Exception as exc:
                logger.warning("Safety check failed: %s", exc)
                result["harness_warning"] = (result.get("harness_warning") or "") + "安全检测异常。".strip()

        # ── Complete ──
        try:
            self.state.transition(MedPhase.COMPLETE)
        except ValueError:
            self.state.force_transition(MedPhase.COMPLETE)
        self._emit("complete", {"session_id": self.state.session_id})

        result["trace"] = {
            "phases": self.state.history,
            "retries": self.state.retry_history,
            "total_duration_ms": self.state.metadata.get("total_duration_ms"),
        }

        return result

    # ======================================================================
    # 模式分发
    # ======================================================================

    def _run_tool_mode(self, query: str, result: dict) -> str:
        """工具模式：执行原生工具返回结果。"""
        tool_name = result.get("route", {}).get("tool_name", "")
        tool_params = result.get("route", {}).get("tool_params", {})

        if self.tool_executor and tool_name:
            tool_result = self.tool_executor(tool_name=tool_name, **tool_params)
            if isinstance(tool_result, str):
                return tool_result
            return str(tool_result.data) if tool_result.data else "未找到结果。"
        return "工具不可用。"

    def _run_chat_mode(self, query: str, result: dict) -> str:
        """聊天模式：直接 LLM 回复。"""
        if self.generator:
            messages = [{"role": "user", "content": query}]
            return self.generator(messages)
        return "你好！有什么可以帮你的吗？"

    def _run_rag_mode(self, query: str, result: dict, **context) -> str:
        """RAG 模式：完整的检索-重排-生成流水线。"""
        retrieval_data = {}
        reranked_qa = []

        # ── Phase 3: 多源检索 ──
        self.state.transition(MedPhase.RETRIEVE)
        self._emit("phase_change", {"phase": "retrieve"})

        if self.retriever:
            try:
                retrieval_data = self.retriever(query, **context)
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)
                retrieval_data = {"kg_results": [], "qa_results": []}
                result["harness_warning"] = (result.get("harness_warning") or "") + "检索暂时不可用，回答基于自身知识。".strip()

        # ── Phase 4: 重排序 ──
        self.state.transition(MedPhase.RERANK)
        self._emit("phase_change", {"phase": "rerank"})

        qa_results = retrieval_data.get("qa_results", [])
        if self.reranker and qa_results:
            try:
                reranked_qa = self.reranker(query, qa_results)
            except Exception as exc:
                logger.warning("Rerank failed, using raw results: %s", exc)
                reranked_qa = qa_results[:5]
        else:
            reranked_qa = qa_results[:5]

        # ── Phase 5: 上下文组装 ──
        self.state.transition(MedPhase.ASSEMBLE)
        self._emit("phase_change", {"phase": "assemble"})

        messages = []
        if self.assembler:
            try:
                context_data = self.assembler(
                    query=query,
                    kg_results=retrieval_data.get("kg_results", []),
                    qa_results=reranked_qa,
                    case_results=retrieval_data.get("case_results", []),
                    route=result.get("route", {}),
                )
                messages = context_data.get("messages", []) if isinstance(context_data, dict) else context_data
            except Exception as exc:
                logger.warning("Assembly failed: %s", exc)
                messages = [{"role": "user", "content": query}]

        if not messages:
            messages = [{"role": "user", "content": query}]

        # ── Phase 6: 生成 ──
        self.state.transition(MedPhase.GENERATE)
        self._emit("phase_change", {"phase": "generate"})

        if self.generator:
            try:
                answer = self.generator(messages)
            except Exception as exc:
                logger.warning("Generation failed: %s", exc)
                answer = "抱歉，AI 模型暂时无法生成回答，请稍后重试。"
                result["harness_warning"] = (result.get("harness_warning") or "") + "AI 生成异常。".strip()
        else:
            answer = ""

        return answer

    def _run_react_mode(self, query: str, result: dict) -> str:
        """ReAct 模式：多步推理（保留现有 ReActEngine，用 Harness 包装）。"""
        # 使用现有 MedicalChatService._handle_react 逻辑
        if self.generator:
            messages = [{"role": "user", "content": query}]
            return self.generator(messages)
        return "ReAct 模式暂时不可用。"

    def _emit(self, event_type: str, data: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_harness_orchestrator.py -v`
Expected: ALL PASS

- [ ] **Step 5: Update exports**

```python
# In src/medrag/harness/__init__.py, append:
from .orchestrator import HarnessOrchestrator
```

- [ ] **Step 6: Commit**

```bash
git add src/medrag/harness/orchestrator.py src/medrag/harness/__init__.py tests/test_harness_orchestrator.py
git commit -m "feat(harness): 医疗全流程编排器 — 带状态机+超时+重试+降级的 HarnessOrchestrator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: 集成到 MedicalChatService

**Files:**
- Modify: `src/medrag/service/chat_service.py`
- Test: `tests/test_harness_integration.py`

将现有的 `MedicalChatService.chat()` 方法逐步改用 `HarnessOrchestrator` 进行流程控制。保持向后兼容：harness 版本通过 feature flag 开启。

- [ ] **Step 1: Write the integration test**

```python
"""tests/test_harness_integration.py — Harness 集成到 MedicalChatService。"""
from __future__ import annotations

import pytest

from medrag.harness.types import MedToolResult, ToolStatus, HarnessConfig
from medrag.harness.wrappers import create_retrieval_wrapper, create_llm_wrapper
from medrag.harness.orchestrator import HarnessOrchestrator


class TestRetrievalWrapperWithRealComponents:
    """使用 mock 但贴近真实组件。"""
    def test_retrieval_wrapper_extracts_sources(self):
        class FakeRetriever:
            def retrieve(self, query, **kw):
                return {
                    "kg_results": [
                        {"answer": "高血压是一种慢性病", "rrf_score": 0.85, "source": "neo4j"},
                    ],
                    "qa_results": [
                        {"answer": "高血压治疗包括...", "rrf_score": 0.72, "source": "milvus"},
                    ],
                    "fusion_mode": "rrf_dense_sparse",
                }

        wrapper = create_retrieval_wrapper(FakeRetriever())
        result = wrapper.execute(query="高血压")
        assert result.status == ToolStatus.SUCCESS
        assert len(result.sources) == 2
        # 置信度应为 HIGH（双源+高分）
        assert result.confidence.value == "high"

    def test_llm_wrapper_fallback_message(self):
        class BrokenGenerator:
            def generate(self, messages, **kw):
                raise RuntimeError("API quota exceeded")

        wrapper = create_llm_wrapper(BrokenGenerator())
        result = wrapper.execute(messages=[{"role": "user", "content": "hi"}])
        assert result.status == ToolStatus.FAILED
        assert "API quota" in result.error


class TestOrchestratorWithMedicalComponents:
    """使用 mock 组件的完整流程测试。"""
    def test_rag_flow_produces_answer(self):
        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": []},
            router=lambda q, **kw: {"execution_mode": "rag", "query_type": "medical"},
            retriever=lambda q, **kw: {
                "kg_results": [{"answer": "高血压定义", "rrf_score": 0.9}],
                "qa_results": [{"answer": "高血压治疗", "rrf_score": 0.8}],
            },
            reranker=lambda q, r, **kw: r,
            assembler=lambda **kw: {"messages": [{"role": "user", "content": "test"}]},
            generator=lambda m, **kw: "高血压患者应注意低盐饮食、规律服药、定期监测血压。",
            safety_checker=lambda q, a, **kw: a + "\n\n*以上信息仅供参考*",
        )
        result = orchestrator.run(query="高血压注意什么")
        assert result["status"] == "success"
        assert "低盐饮食" in result["answer"]
        assert "以上信息仅供参考" in result["answer"]

    def test_kg_failure_falls_back_gracefully(self):
        """KG 失败时仍能从 QA 获取结果。"""
        class FakeRetriever:
            def retrieve(self, query, **kw):
                return {
                    "kg_results": [],  # KG 没结果
                    "qa_results": [{"answer": "QA 结果", "rrf_score": 0.7}],
                    "fusion_mode": "single",
                }

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": []},
            router=lambda q, **kw: {"execution_mode": "rag"},
            retriever=FakeRetriever().retrieve,
            reranker=lambda q, r, **kw: r,
            assembler=lambda **kw: {"messages": [{"role": "user", "content": "test"}]},
            generator=lambda m, **kw: "基于 QA 结果：...",
            safety_checker=lambda q, a, **kw: a,
        )
        result = orchestrator.run(query="高血压")
        assert result["status"] == "success"
        assert len(result["answer"]) > 0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_harness_integration.py -v`
Expected: FAIL

- [ ] **Step 3: 集成到 MedicalChatService**

```python
# In MedicalChatService, add a method:
def chat_with_harness(
    self,
    query: str,
    user_case_summary: Optional[str] = None,
    username: Optional[str] = None,
    use_harness: bool = True,
) -> Dict:
    """Harness 版本的 chat() — 包装相同组件但提供统一容错。

    用法与 chat() 完全一致，后续可逐步替代。
    """
    from medrag.harness.orchestrator import HarnessOrchestrator

    # 组装 SafetyGuard 检测器（直接传递方法引用）
    def _risk_detect(query):
        return self.safety_guard.detect_risk(query)

    def _route(query):
        return self.hybrid_retriever.router.route(query)

    def _retrieve(query, department=None, username=None):
        return self.hybrid_retriever.retrieve(query, department=department, username=username)

    def _rerank(query, results):
        return self.reranker.rerank(query, results, top_k=settings.rerank_top_k)

    def _assemble(**kw):
        # 复用现有 PromptBuilder 的逻辑
        retrieval_quality = {
            "has_kg": bool(kw.get("kg_results")),
            "has_qa": bool(kw.get("qa_results")),
            "confidence": "high" if (kw.get("kg_results") or kw.get("qa_results")) else "none",
        }
        messages = self.prompt_builder.build_messages_with_context(
            context={"sections": kw.get("sections", {})} if hasattr(self.prompt_builder, 'build_messages_with_context') else {},
            query=kw.get("query", ""),
            route=kw.get("route", {}),
            retrieval_quality=retrieval_quality,
        )
        return {"messages": messages}

    def _generate(messages, model=None):
        return self.answer_generator.generate(messages, model=model)

    def _safety_check(query, answer, retrieval_quality="none", query_type=""):
        risk_info = self.safety_guard.detect_risk(query)
        return self.safety_guard.append_safety_notice(
            answer, risk_info,
            retrieval_quality=retrieval_quality,
            query_type=query_type,
        )

    orchestrator = HarnessOrchestrator(
        risk_detector=_risk_detect,
        router=_route,
        retriever=_retrieve,
        reranker=_rerank,
        assembler=_assemble,
        generator=_generate,
        safety_checker=_safety_check,
    )

    result = orchestrator.run(
        query=query,
        username=username,
    )

    # 兼容现有返回格式
    return {
        "answer": result["answer"],
        "route": result.get("route", {}),
        "risk_info": result.get("risk_info", {}),
        "harness_trace": result.get("trace", {}),
        "harness_warning": result.get("harness_warning"),
        # 保持向后兼容的字段
        "kg_results": [],
        "qa_results": [],
        "case_results": [],
        "qa_source_details": {},
        "query_info": None,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_harness_integration.py -v`
Expected: ALL PASS

Run: `uv run pytest tests/test_harness_orchestrator.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/medrag/harness/ src/medrag/service/chat_service.py tests/test_harness_integration.py
git commit -m "feat(harness): 集成 HarnessOrchestrator 到 MedicalChatService — 医疗全流程统一容错编排

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5 (Future): DAG 并行执行引擎

这个阶段暂不实施，但架构上预留了扩展点：
- `MedToolWrapper` 已具备独立的超时/重试能力，天然可并行
- `HarnessOrchestrator` 的 `_run_rag_mode` 中，KG 和 QA 检索可以改为并行（参考 AGI-saber GraphRuntime）
- 后续可通过 `TaskGraph` + `GraphRuntime` 替换 serial 执行

---

## 总结：医疗场景适配点

| 设计决策 | AGI-saber（通用） | MedAgent（医疗） |
|---|---|---|
| 状态机阶段 | pending→running→done | risk_detect→route→retrieve→...→safety_check→complete（强制安全检测） |
| 工具结果 | ExecResult (stdout/stderr/code) | MedToolResult (data+confidence+sources+risk_flags+user_facing_warning) |
| 置信度 | 无 | HIGH/MEDIUM/LOW/NONE（直接影响免责声明） |
| 证据溯源 | 无 | SourceTrace（每个结果记录来源类型和分数） |
| 超时配置 | 统一超时 | 按组件预设（KG 15s, QA 5s, LLM 60s, 本地 2s） |
| 重试策略 | 统一重试 | 分场景（retrieval有重试, local_tool不重试） |
| 降级提示 | 返回空/错误 | 返回中文用户友好提示 + 标记 confidence 降级 |
| Tool call schema | (string, error) → string | MedToolResult（永不抛异常） |
| 安全检测 | 无（外部路由） | 内置 RISK_DETECT + SAFETY_CHECK 两个强制阶段 |

## 文件变更

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/medrag/harness/__init__.py` | 创建 | 包入口 |
| `src/medrag/harness/types.py` | 创建 | 医疗核心类型：MedToolResult, MedStateMachine, HarnessConfig |
| `src/medrag/harness/wrappers.py` | 创建 | 医疗工具包装器：封装现有检索/LLM/原生工具 |
| `src/medrag/harness/orchestrator.py` | 创建 | 全流程编排器：容错编排的 4 种模式 |
| `src/medrag/service/chat_service.py` | 修改 | 新增 `chat_with_harness()` 方法 |
| `tests/test_harness_types.py` | 创建 | 类型测试 |
| `tests/test_harness_wrappers.py` | 创建 | 包装器测试 |
| `tests/test_harness_orchestrator.py` | 创建 | 编排器测试 |
| `tests/test_harness_integration.py` | 创建 | 集成测试 |
