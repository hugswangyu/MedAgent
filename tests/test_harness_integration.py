"""tests/test_harness_integration.py — Harness 集成到 MedicalChatService。"""
from __future__ import annotations

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
            safety_checker=lambda query, answer, **kw: answer + "\n\n*以上信息仅供参考*",
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
                    "kg_results": [],
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
            safety_checker=lambda query, answer, **kw: answer,
        )
        result = orchestrator.run(query="高血压")
        assert result["status"] == "success"
        assert len(result["answer"]) > 0
