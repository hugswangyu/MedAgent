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
            safety_checker=lambda query, answer, **kw: answer + "\n\n[紧急提示] 请立即就医",
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
            safety_checker=lambda query, answer, **kw: answer + "\n\n*信息有限*",
        )
        result = orchestrator.run(query="高血压")
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
            safety_checker=lambda query, answer, **kw: answer + "\n\n*仅供参考*",
            on_event=on_event,
        )
        orchestrator.run(query="高血压")
        event_types = [e[0] for e in events]
        assert "phase_change" in event_types
        assert "complete" in event_types
