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

    def test_failed_result_propagates_warning(self):
        r = MedToolResult.failed(
            tool_name="kg_search",
            error="Neo4j connection refused",
            user_facing_warning="知识图谱暂时不可用，回答可能不完整。",
        )
        assert r.status == ToolStatus.FAILED
        assert r.user_facing_warning

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
        assert bool(r) is True

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
        assert c.tool_timeouts["kg_search"] == 15000
        assert c.tool_timeouts["qa_search"] == 5000
        assert c.tool_timeouts["llm_generate"] == 60000
        assert c.tool_timeouts["dosage_calculator"] == 2000

    def test_retrieval_retry_policy(self):
        c = HarnessConfig()
        assert c.retry_policies["retrieval"]["max_retries"] == 2
        assert c.retry_policies["local_tool"]["max_retries"] == 0

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
        sm.force_transition(MedPhase.GENERATE)
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
