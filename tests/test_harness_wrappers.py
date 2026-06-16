"""tests/test_harness_wrappers.py — 医疗工具包装器。"""
from __future__ import annotations

import pytest

from medrag.harness.types import (
    MedToolResult, ToolStatus, Confidence, SourceTrace, HarnessConfig,
)
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
        def slow(**kw):
            import time
            time.sleep(10)
            return "done"

        wrapper = MedToolWrapper(
            name="kg_search", category="retrieval",
            executor=slow,
        )
        config = HarnessConfig(override_timeouts={"kg_search": 1})
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
        def ok(**kw):
            return {"answer": "data"}

        wrapper = MedToolWrapper(name="test", category="retrieval", executor=ok)
        result = wrapper.execute()
        assert result.confidence == Confidence.HIGH


class TestCreateRetrievalWrapper:
    def test_wraps_hybrid_retriever(self):
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
        kg_traces = [s for s in result.sources if s.source_type == "kg"]
        assert len(kg_traces) >= 1


class TestCreateLLMWrapper:
    def test_wraps_answer_generator(self):
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
