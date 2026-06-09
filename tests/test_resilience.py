"""架构弹性与优雅降级测试。"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from medrag.rag.answer_generator import AnswerGenerator
from medrag.vectors.embedding import EmbeddingModel


# ============================================================================
# EmbeddingModel 故障测试
# ============================================================================


class TestEmbeddingModelFailure:
    """EmbeddingModel init 失败时优雅降级。"""

    @patch("medrag.vectors.embedding.SentenceTransformer", side_effect=RuntimeError("OOM"))
    def test_init_failure_sets_unavailable(self, mock_st):
        model = EmbeddingModel("BAAI/bge-small-zh-v1.5")
        assert model._available is False
        assert model.model is None
        assert model.embedding_dim == 0

    @patch("medrag.vectors.embedding.SentenceTransformer", side_effect=RuntimeError("OOM"))
    def test_encode_returns_empty_on_unavailable(self, mock_st):
        model = EmbeddingModel("BAAI/bge-small-zh-v1.5")
        result = model.encode(["test query"], is_query=True)
        assert result == []

    @patch("medrag.vectors.embedding.SentenceTransformer", side_effect=RuntimeError("OOM"))
    def test_encode_one_returns_empty_on_unavailable(self, mock_st):
        model = EmbeddingModel("BAAI/bge-small-zh-v1.5")
        result = model.encode_one("test query", is_query=True)
        assert result == []


# ============================================================================
# QARetriever 故障测试
# ============================================================================


class TestQARetrieverFailure:
    """QARetriever 子组件故障时优雅降级。"""

    @patch("medrag.vectors.qa_retriever.EmbeddingModel", side_effect=RuntimeError("OOM"))
    @patch("medrag.vectors.qa_retriever.MilvusClientWrapper")
    def test_init_handles_embedding_failure(self, mock_milvus, mock_emb):
        from medrag.vectors.qa_retriever import QARetriever
        retriever = QARetriever()
        assert retriever.embedding_model is None

    @patch("medrag.vectors.qa_retriever.EmbeddingModel")
    @patch("medrag.vectors.qa_retriever.MilvusClientWrapper", side_effect=ConnectionError("Milvus down"))
    def test_init_handles_milvus_failure(self, mock_milvus, mock_emb):
        from medrag.vectors.qa_retriever import QARetriever
        retriever = QARetriever()
        assert retriever.collection is None
        assert retriever._available is False

    def test_search_raises_when_unavailable(self):
        from medrag.vectors.qa_retriever import QARetriever
        retriever = QARetriever.__new__(QARetriever)
        retriever.collection = None
        retriever._available = False
        retriever.embedding_model = None
        with pytest.raises(RuntimeError, match="QARetriever unavailable"):
            retriever.search("test")


# ============================================================================
# AnswerGenerator 缓存测试
# ============================================================================


class TestAnswerGeneratorCache:
    """LLM 响应缓存。"""

    def test_cache_hit_returns_cached(self):
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "test-model"
        mock_provider.client = MagicMock()

        gen = AnswerGenerator(llm_provider=mock_provider, cache_max_size=10, cache_ttl_seconds=3600)

        cache_key = gen._make_cache_key("test-model", [{"role": "user", "content": "hello"}])
        gen._cache[cache_key] = (datetime.now(), "cached response")

        result = gen.generate("hello")
        assert result == "cached response"
        mock_provider.client.chat.completions.create.assert_not_called()

    def test_cache_miss_calls_llm(self):
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "test-model"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "fresh response"
        mock_provider.client.chat.completions.create.return_value = mock_response

        gen = AnswerGenerator(llm_provider=mock_provider)
        result = gen.generate("hello")
        assert result == "fresh response"
        mock_provider.client.chat.completions.create.assert_called_once()

    def test_cache_ttl_expiry(self):
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "test-model"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "fresh"
        mock_provider.client.chat.completions.create.return_value = mock_response

        gen = AnswerGenerator(llm_provider=mock_provider, cache_ttl_seconds=1)

        cache_key = gen._make_cache_key("test-model", [{"role": "user", "content": "hello"}])
        # Entry from 2h ago - well past 1s TTL
        gen._cache[cache_key] = (datetime.now() - timedelta(hours=2), "stale")

        result = gen.generate("hello")
        assert result == "fresh"

    def test_cache_eviction_fifo(self):
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "test-model"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "answer"
        mock_provider.client.chat.completions.create.return_value = mock_response

        gen = AnswerGenerator(llm_provider=mock_provider, cache_max_size=2, cache_ttl_seconds=3600)
        gen.generate("q1")
        gen.generate("q2")
        gen.generate("q3")

        # q1 should be evicted (FIFO), q2 and q3 should be in cache
        assert len(gen._cache) == 2
        # Verify q1's cache key is gone
        k1 = gen._make_cache_key("test-model", [{"role": "user", "content": "q1"}])
        assert gen._cache_get(k1) is None


# ============================================================================
# AnswerGenerator 回退测试
# ============================================================================


class TestAnswerGeneratorFallback:
    """主 LLM 失败时回退到 Ollama。"""

    @patch("medrag.rag.answer_generator.get_llm_provider")
    def test_primary_failure_falls_back_to_ollama(self, mock_get_provider):
        primary = MagicMock()
        primary.name = "deepseek"
        primary.default_model = "deepseek-chat"
        primary.client.chat.completions.create.side_effect = Exception("API error")

        fallback = MagicMock()
        fallback.name = "ollama"
        fallback.default_model = "qwen3.5:9b"
        fallback_response = MagicMock()
        fallback_response.choices[0].message.content = "fallback answer"
        fallback.client.chat.completions.create.return_value = fallback_response

        def side_effect(name=None):
            if name and name.strip().lower() == "ollama":
                return fallback
            return primary

        mock_get_provider.side_effect = side_effect

        gen = AnswerGenerator(llm_provider=primary)
        result = gen.generate("test query")
        assert result == "fallback answer"
        primary.client.chat.completions.create.assert_called_once()
        fallback.client.chat.completions.create.assert_called_once()

    @patch("medrag.rag.answer_generator.get_llm_provider")
    def test_both_fail_return_error_message(self, mock_get_provider):
        primary = MagicMock()
        primary.name = "deepseek"
        primary.default_model = "deepseek-chat"
        primary.client.chat.completions.create.side_effect = Exception("API error")

        def side_effect(name=None):
            fb = MagicMock()
            fb.name = "ollama"
            fb.default_model = "qwen3.5:9b"
            fb.client.chat.completions.create.side_effect = Exception("Ollama down")
            return fb

        mock_get_provider.side_effect = side_effect

        gen = AnswerGenerator(llm_provider=primary)
        result = gen.generate("test query")
        assert "出错" in result

    def test_stream_fallback_on_connection_error(self):
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "test-model"
        mock_provider.client.chat.completions.create.side_effect = Exception("Connection failed")

        gen = AnswerGenerator(llm_provider=mock_provider, cache_max_size=10, cache_ttl_seconds=3600)

        # Without Ollama available, should yield error
        # _try_fallback will fail because get_llm_provider("ollama") raises ValueError
        tokens = list(gen.generate_stream("hello"))
        assert any("错误" in t for t in tokens)


# ============================================================================
# 健康注册表测试
# ============================================================================


class TestHealthRegistry:
    """健康状态追踪。"""

    def setup_method(self):
        import medrag.infrastructure.health as h
        h._status.clear()
        h._errors.clear()

    def test_initial_state_empty(self):
        from medrag.infrastructure.health import get_summary
        info = get_summary()
        assert info["status"] == "ok"
        assert info["components"] == {}

    def test_report_ok(self):
        from medrag.infrastructure.health import report_ok, get_summary
        report_ok("milvus")
        info = get_summary()
        assert info["components"]["milvus"]["status"] == "ok"

    def test_report_down(self):
        from medrag.infrastructure.health import report_down, get_summary
        report_down("neo4j", "connection refused")
        info = get_summary()
        assert info["components"]["neo4j"]["status"] == "down"
        assert "connection refused" in info["components"]["neo4j"]["error"]
        assert info["status"] == "down"

    def test_report_degraded(self):
        from medrag.infrastructure.health import report_degraded, report_ok, get_summary
        report_ok("milvus")
        report_degraded("llm", "rate limited")
        info = get_summary()
        assert info["status"] == "degraded"
        assert info["components"]["llm"]["status"] == "degraded"

    def test_down_overrides_degraded(self):
        from medrag.infrastructure.health import report_degraded, report_down, get_summary
        report_degraded("llm", "slow")
        report_down("neo4j", "down")
        info = get_summary()
        assert info["status"] == "down"
        assert info["components"]["llm"]["status"] == "degraded"
        assert info["components"]["neo4j"]["status"] == "down"


# ============================================================================
# HybridRetriever 降级测试
# ============================================================================


class TestHybridRetrieverDegradation:
    """HybridRetriever 组件故障时降级。"""

    def test_both_qa_and_es_fail_return_empty(self):
        mock_qa = MagicMock()
        mock_qa.search.side_effect = RuntimeError("Milvus down")
        mock_es = MagicMock()
        mock_es.search.return_value = []

        from medrag.retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever(qa_retriever=mock_qa, es_retriever=mock_es)
        hybrid.router = MagicMock()
        hybrid.router.route.return_value = {"use_kg": False, "use_qa": True, "needs_case_context": False}

        result = hybrid.retrieve("test query")
        assert result["qa_results"] == []

    def test_qa_fails_es_works_returns_es_only(self):
        mock_qa = MagicMock()
        mock_qa.search.side_effect = RuntimeError("Milvus down")
        mock_es = MagicMock()
        mock_es.search.return_value = [
            {"id": "1", "answer": "ES result", "score": 0.8, "title": "", "question": "", "text": ""}
        ]

        from medrag.retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever(qa_retriever=mock_qa, es_retriever=mock_es)
        hybrid.router = MagicMock()
        hybrid.router.route.return_value = {"use_kg": False, "use_qa": True, "needs_case_context": False}

        result = hybrid.retrieve("test query")
        assert len(result["qa_results"]) == 1
        assert result["qa_results"][0]["answer"] == "ES result"
        assert result["fusion_mode"] == "single"
