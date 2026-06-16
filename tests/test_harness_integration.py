"""tests/test_harness_integration.py — Harness 集成到 MedicalChatService（ReAct 架构）。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from medrag.harness.types import MedToolResult, ToolStatus, HarnessConfig
from medrag.harness.wrappers import create_retrieval_wrapper, create_llm_wrapper
from medrag.harness.orchestrator import HarnessOrchestrator


class TestRetrievalWrapperWithRealComponents:
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
    def test_react_flow_produces_answer_with_safety_check(self):
        """ReAct 循环 + safety_check 完整流程。"""

        def fake_builder(query, route):
            engine = MagicMock()
            engine.run.return_value = {
                "answer": "高血压患者应注意低盐饮食、规律服药、定期监测血压。",
                "steps": [
                    {"step": 1, "action": "retrieve_knowledge",
                     "input": {"query": "高血压"}, "observation": "...", "thought": "需要检索知识"},
                ],
                "tool_results": {"retrieve_knowledge": ["【知识图谱结果】\n- 高血压是一种慢性病"]},
            }
            return engine

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": False, "risk_keywords": []},
            router=lambda q, **kw: {"query_type": "disease_fact"},
            react_engine_builder=fake_builder,
            safety_checker=lambda query, answer, **kw: answer + "\n\n*以上信息仅供参考*",
        )
        result = orchestrator.run(query="高血压注意什么")
        assert result["status"] == "success"
        assert "低盐饮食" in result["answer"]
        assert "以上信息仅供参考" in result["answer"]
        assert len(result["react_trace"]["steps"]) == 1
        assert result["react_trace"]["steps"][0]["action"] == "retrieve_knowledge"

    def test_react_loop_with_tool_results_in_trace(self):
        """工具调用结果应在 react_trace 中体现。"""

        def fake_builder(query, route):
            engine = MagicMock()
            engine.run.return_value = {
                "answer": "基于检索结果：...",
                "steps": [
                    {"step": 1, "action": "retrieve_knowledge",
                     "input": {"query": "高血压"}, "observation": "...", "thought": ""},
                ],
                "tool_results": {
                    "retrieve_knowledge": [
                        "【知识图谱结果】\n- 高血压是一种慢性病\n【相似问答结果】\n- 高血压治疗包括..."
                    ],
                },
            }
            return engine

        orchestrator = HarnessOrchestrator(
            react_engine_builder=fake_builder,
        )
        result = orchestrator.run(query="高血压")
        assert result["react_trace"]["tool_results"]["retrieve_knowledge"][0].startswith("【知识图谱结果】")

    def test_risk_detect_blocks_downstream_on_failure(self):
        """风险检测不应阻断下游流程，但信息应被记录。"""

        def fake_builder(query, route):
            engine = MagicMock()
            engine.run.return_value = {
                "answer": "正常回答",
                "steps": [],
                "tool_results": {},
            }
            return engine

        orchestrator = HarnessOrchestrator(
            risk_detector=lambda q, **kw: {"has_risk": True, "risk_keywords": ["紧急"], "level": "high"},
            react_engine_builder=fake_builder,
            safety_checker=lambda query, answer, **kw: answer + "\n\n*请尽快就医*",
        )
        result = orchestrator.run(query="紧急情况")
        assert result["status"] == "success"
        assert result["risk_info"]["has_risk"] is True
        assert "请尽快就医" in result["answer"]
