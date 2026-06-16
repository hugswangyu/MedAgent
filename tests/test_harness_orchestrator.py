"""tests/test_harness_orchestrator.py — HarnessOrchestrator 测试（ReAct 统一架构）。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from medrag.harness.types import (
    MedToolResult, ToolStatus, Confidence, MedPhase, HarnessConfig,
)
from medrag.harness.orchestrator import HarnessOrchestrator


class TestHarnessOrchestrator:
    def test_react_loop_produces_answer(self):
        """ReAct 循环正常返回答案，safety_check 追加免责声明。"""

        def fake_builder(query, route):
            engine = MagicMock()
            engine.run.return_value = {
                "answer": "高血压患者应注意低盐饮食",
                "steps": [{"step": 1, "action": "retrieve_knowledge", "input": {"query": "高血压"}, "observation": "...", "thought": "需要检索"}],
                "tool_results": {"retrieve_knowledge": ["..."]},
            }
            return engine

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": [], "level": "none"},
            router=lambda q, **kw: {"query_type": "disease_fact"},
            react_engine_builder=fake_builder,
            safety_checker=lambda query, answer, **kw: answer + "\n\n*仅供参考*",
        )
        result = orchestrator.run(query="高血压注意什么")
        assert result["status"] == "success"
        assert "低盐饮食" in result["answer"]
        assert "仅供参考" in result["answer"]
        assert result["react_trace"] is not None
        assert len(result["react_trace"]["steps"]) == 1

    def test_risk_detect_triggers_warning(self):
        """风险检测结果应在 risk_info 中体现。"""
        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": True, "risk_keywords": ["胸痛"], "level": "high"},
            react_engine_builder=lambda q, r: MagicMock(
                **{"run.return_value": {"answer": "请及时就医", "steps": [], "tool_results": {}}}
            ),
        )
        result = orchestrator.run(query="胸痛怎么办")
        assert result["risk_info"]["has_risk"] is True
        assert "胸痛" in result["risk_info"]["risk_keywords"]

    def test_react_loop_failure_falls_back(self):
        """ReAct 循环异常时应有降级提示。"""
        def broken_builder(query, route):
            raise RuntimeError("LLM unavailable")

        orchestrator = HarnessOrchestrator(
            react_engine_builder=broken_builder,
        )
        result = orchestrator.run(query="高血压")
        assert result["status"] == "failed"
        assert "harness_warning" in result

    def test_no_react_engine_builder(self):
        """未配置 react_engine_builder 时应有提示。"""
        orchestrator = HarnessOrchestrator()
        result = orchestrator.run(query="测试")
        assert "尚未配置推理引擎" in result["answer"]

    def test_safety_check_appends_notice(self):
        """SAFETY_CHECK 阶段追加免责声明。"""
        def fake_builder(query, route):
            engine = MagicMock()
            engine.run.return_value = {"answer": "测试回答", "steps": [], "tool_results": {}}
            return engine

        orchestrator = HarnessOrchestrator(
            react_engine_builder=fake_builder,
            safety_checker=lambda query, answer, **kw: answer + "\n\n*仅供医学参考*",
        )
        result = orchestrator.run(query="测试")
        assert "仅供医学参考" in result["answer"]

    def test_phases_trace_is_recorded(self):
        """执行阶段历史应完整记录。"""
        def fake_builder(query, route):
            engine = MagicMock()
            engine.run.return_value = {"answer": "回答", "steps": [], "tool_results": {}}
            return engine

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": [], "level": "none"},
            router=lambda q, **kw: {"query_type": "disease_fact"},
            react_engine_builder=fake_builder,
            safety_checker=lambda query, answer, **kw: answer,
        )
        result = orchestrator.run(query="测试")
        phases = [p["phase"] for p in result["trace"]["phases"]]
        assert "risk_detect" in phases
        assert "route" in phases
        assert "react_loop" in phases
        assert "safety_check" in phases
        assert "complete" in phases

    def test_event_emission(self):
        """事件通知应包含 start、phase_change 和 complete。"""
        events = []

        def on_event(event_type, data):
            events.append((event_type, data))

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": [], "level": "none"},
            router=lambda q, **kw: {"query_type": "disease_fact"},
            react_engine_builder=lambda q, r: MagicMock(
                **{"run.return_value": {"answer": "回答", "steps": [], "tool_results": {}}}
            ),
            safety_checker=lambda query, answer, **kw: answer,
            on_event=on_event,
        )
        orchestrator.run(query="高血压")
        event_types = [e[0] for e in events]
        assert "start" in event_types
        assert "phase_change" in event_types
        assert "complete" in event_types
