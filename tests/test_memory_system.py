"""记忆系统测试：MemorySystem 单元 + chat_service 集成。"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from medrag.memory import MemorySystem, get_memory_system
from medrag.memory.short_term import ConversationMessage
from medrag.memory.types import ConsolidationConfig, MemoryItem, RecallFilter


# ============================================================================
# 辅助方法
# ============================================================================

def _make_embedding(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float64)


# ============================================================================
# MemorySystem 单元测试
# ============================================================================


class TestMemorySystemBasics:
    """基本 CRUD + context 构建。"""

    def test_store_and_recall(self):
        ms = MemorySystem(max_turns=3)
        ms.remember("患者有高血压病史", importance=0.8)
        assert ms.stats["ltm_count"] == 1
        results = ms.recall("高血压")
        assert len(results) >= 1
        assert "高血压" in results[0].content

    def test_add_message_user_stores_preference(self):
        ms = MemorySystem()
        ms.add_message("user", "我叫张三")
        assert ms.preferences.get("姓名") == "张三"
        results = ms.recall("张三")
        assert any("张三" in r.content for r in results)

    def test_add_message_classifies_medical_fact(self):
        ms = MemorySystem()
        ms.add_message("user", "我对青霉素过敏")
        results = ms.recall("过敏")
        assert any("青霉素" in r.content for r in results)

    def test_add_message_general_skips_ltm(self):
        ms = MemorySystem()
        ms.add_message("user", "今天天气真好")
        assert ms.stats["ltm_count"] == 0

    def test_store_assistant_reply(self):
        ms = MemorySystem()
        ms.store_assistant_reply("患者有高血压需要注意饮食")
        results = ms.recall("高血压")
        assert any("高血压" in r.content for r in results)

    def test_build_context_with_all_layers(self):
        ms = MemorySystem(max_turns=5)
        ms.add_message("user", "我叫张三")
        ms.add_message("user", "我对青霉素过敏")
        ms.add_message("user", "最近总是头痛")
        ms.store_assistant_reply("建议多休息")
        ctx = ms.build_context("张三")
        assert "【用户偏好】" in ctx
        assert "张三" in ctx
        assert "【对话历史】" in ctx

    def test_build_context_empty(self):
        ms = MemorySystem()
        ctx = ms.build_context("你好")
        assert ctx == ""

    def test_recall_with_embedding(self):
        ms = MemorySystem()
        emb = _make_embedding()
        ms.remember("高血压需要低盐饮食", importance=0.9, embedding=emb)
        results = ms.recall(query_embedding=emb, top_k=1)
        assert len(results) == 1

    def test_clear_resets_all(self):
        ms = MemorySystem()
        ms.add_message("user", "我叫张三")
        ms.clear()
        assert ms.stats["ltm_count"] == 0
        assert ms.stats["preferences"] == 0
        assert ms.stats["stm_count"] == 0

    def test_singleton(self):
        a = get_memory_system()
        b = get_memory_system()
        assert a is b

    def test_consolidate_now(self):
        cfg = ConsolidationConfig(
            similarity_threshold=0.95,
            dedup_threshold=0.99,
            ttl_days=365,
            decay_rate=1.0,
            min_importance=0.1,
            trigger_interval=100,
        )
        ms = MemorySystem(consolidation=cfg)
        ms.remember("高血压注意事项", importance=0.5)
        ms.remember("高血压注意事项", importance=0.6)
        result = ms.consolidate_now()
        assert result is not None


class TestMemorySystemSTM:
    """短期记忆窗口测试。"""

    def test_sliding_window(self):
        ms = MemorySystem(max_turns=2)
        for i in range(10):
            ms.add_message("user", f"消息{i}")
        ctx = ms.build_context("test")
        # 最多 2 轮 = 4 条 STM 消息
        for i in [8, 9]:
            assert f"消息{i}" in ctx
        for i in [0, 1]:
            assert f"消息{i}" not in ctx

    def test_stm_stats(self):
        ms = MemorySystem(max_turns=3)
        ms.add_message("user", "a")
        ms.add_message("assistant", "b")
        assert ms.stats["stm_count"] == 2
        assert ms.stats["msg_count"] == 2


class TestMemorySystemRecallFilter:
    """召回过滤测试。"""

    def test_category_filter(self):
        ms = MemorySystem()
        ms.remember("我叫张三", importance=0.8, category="identity", tags=["name"])
        ms.remember("对青霉素过敏", importance=0.9, category="fact", tags=["medical"])
        results = ms.recall("过敏", filter=RecallFilter(categories=["fact"]))
        assert len(results) >= 1
        assert results[0].category == "fact"

    def test_top_k_limit(self):
        ms = MemorySystem()
        for i in range(10):
            ms.remember(f"记忆项目{i}", importance=0.5)
        results = ms.recall("记忆", filter=RecallFilter(top_k=3, min_score=0.0))
        assert len(results) <= 3


# ============================================================================
# LTM consolidation 测试
# ============================================================================


class TestConsolidation:
    def test_dedup_during_store(self):
        """高相似度存储应触发内联去重。"""
        cfg = ConsolidationConfig(
            dedup_threshold=0.90,
            similarity_threshold=0.80,
            ttl_days=365,
            decay_rate=1.0,
            min_importance=0.1,
            trigger_interval=100,
        )
        ms = MemorySystem(consolidation=cfg)
        emb = _make_embedding()
        ms.remember("高血压注意事项", importance=0.5, embedding=emb)
        added = ms.remember("高血压注意事项", importance=0.7, embedding=emb)
        assert not added  # dedup
        assert ms.stats["ltm_count"] == 1

    def test_low_sim_no_dedup(self):
        """低相似度不应去重。"""
        cfg = ConsolidationConfig(
            dedup_threshold=0.95,
            similarity_threshold=0.80,
            ttl_days=365,
            decay_rate=1.0,
            min_importance=0.1,
            trigger_interval=100,
        )
        ms = MemorySystem(consolidation=cfg)
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ms.remember("高血压", importance=0.5, embedding=e1)
        ms.remember("糖尿病", importance=0.5, embedding=e2)
        assert ms.stats["ltm_count"] == 2


# ============================================================================
# chat_service 集成测试
# ============================================================================


class TestMemoryPersistence:
    """JSON 持久化 save/load 测试。"""

    def test_save_and_load(self, tmp_path):
        persist = tmp_path / "memory.json"
        ms = MemorySystem(persist_path=str(persist))
        ms.remember("高血压病史", importance=0.9, category="fact", tags=["medical"])
        ms.remember("对青霉素过敏", importance=0.8, category="fact", tags=["medical", "allergy"])
        assert ms.stats["ltm_count"] == 2

        # 新建一个 MemorySystem 读取同一文件，应恢复 2 条记忆
        ms2 = MemorySystem(persist_path=str(persist))
        assert ms2.stats["ltm_count"] == 2
        results = ms2.recall("过敏")
        assert any("青霉素" in r.content for r in results)

    def test_persistence_roundtrip_with_embedding(self, tmp_path):
        persist = tmp_path / "memory.json"
        ms = MemorySystem(persist_path=str(persist))
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        ms.remember("高血压注意事项", importance=0.9, embedding=emb)
        ms.remember("糖尿病饮食", importance=0.7)
        assert ms.stats["ltm_count"] == 2

        ms2 = MemorySystem(persist_path=str(persist))
        assert ms2.stats["ltm_count"] == 2
        # 验证 embedding 正确恢复
        items = ms2.long_term.items
        for item in items:
            if "高血压" in item.content:
                assert item.embedding is not None
                assert np.allclose(item.embedding, emb)
                break
        else:
            pytest.fail("高血压 item not found")

    def test_no_persist_does_not_save(self):
        ms = MemorySystem()  # no persist_path
        ms.remember("测试内容", importance=0.5)
        assert ms.stats["ltm_count"] == 1
        # 不传 persist_path 时不应创建文件
        assert ms.long_term._persist_path is None

    def test_load_nonexistent_file(self, tmp_path):
        persist = tmp_path / "nonexistent.json"
        ms = MemorySystem(persist_path=str(persist))
        assert ms.stats["ltm_count"] == 0

    def test_clear_keeps_persistence(self, tmp_path):
        persist = tmp_path / "memory.json"
        ms = MemorySystem(persist_path=str(persist))
        ms.remember("测试", importance=0.5)
        ms.clear()
        assert ms.stats["ltm_count"] == 0
        # 清除后存新数据不应报错
        ms.remember("新测试", importance=0.5)
        assert ms.stats["ltm_count"] == 1


# ============================================================================
# ContextAssembler（Schema-Driven Assembly）
# ============================================================================


class TestContextAssembler:
    """Schema-Driven Assembly 测试。"""

    def test_assemble_basic(self):
        from medrag.memory import ContextAssembler
        a = ContextAssembler(budget=4096)
        a.add("memory", "【长期记忆】\n- 患者有高血压", priority=100)
        a.add("kg", "【知识图谱】\n- 高血压低盐饮食", priority=80)
        result = a.assemble()
        assert "高血压" in result
        assert a.total_tokens > 0

    def test_assemble_empty(self):
        from medrag.memory import ContextAssembler
        a = ContextAssembler(budget=4096)
        assert a.assemble() == ""

    def test_budget_pruning_drops_low_priority(self):
        from medrag.memory import ContextAssembler
        a = ContextAssembler(budget=10)
        a.add("memory", "患者有高血压病史", priority=100)
        a.add("qa", "多喝水休息有助于康复" * 20, priority=70)
        result = a.assemble()
        assert "高血压" in result
        assert "多喝水" not in result
        assert "qa" in a.dropped_slots

    def test_all_slots_under_budget(self):
        from medrag.memory import ContextAssembler
        a = ContextAssembler(budget=9999)
        a.add("memory", "记忆内容", priority=100)
        a.add("kg", "图谱内容", priority=80)
        a.add("qa", "问答内容", priority=70)
        result = a.assemble()
        assert "记忆内容" in result
        assert "图谱内容" in result
        assert "问答内容" in result
        assert not a.dropped_slots

    def test_estimate_tokens(self):
        from medrag.memory.schema import estimate_tokens
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("你好世界") > 0
        assert estimate_tokens("") == 0

    def test_reset(self):
        from medrag.memory import ContextAssembler
        a = ContextAssembler(budget=4096)
        a.add("memory", "内容", priority=100)
        assert a.assemble() != ""
        a.reset()
        assert a.assemble() == ""


class TestChatServiceMemoryIntegration:
    """验证 chat_service 各 handler 正确与 MemorySystem 交互。"""

    @pytest.fixture
    def service(self):
        """创建带 Mock 的 chat_service。"""
        from medrag.service.chat_service import MedicalChatService

        # 注入 MemorySystem
        mem = MemorySystem(max_turns=5)

        # Mock answer_generator
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "这是一个模拟回答。"
        mock_gen.generate_stream.return_value = iter(["这是", "一个", "模拟回答。"])

        # Mock hybrid_retriever.router
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "execution_mode": "rag",
            "query_type": "general_medical_qa",
        }

        # Mock hybrid_retriever
        mock_retriever = MagicMock()
        mock_retriever.router = mock_router
        mock_retriever.retrieve.return_value = {
            "kg_results": [],
            "qa_results": [],
            "case_results": [],
            "qa_source_details": {},
            "query_info": None,
        }

        # Mock tool registry
        mock_tools = MagicMock()
        mock_tools.match.return_value = (None, None)

        # Mock safety_guard
        mock_safety = MagicMock()
        mock_safety.detect_risk.return_value = {"has_risk": False, "risk_keywords": []}
        mock_safety.append_safety_notice.side_effect = lambda answer, risk_info, **kw: answer

        svc = MedicalChatService(
            memory_system=mem,
            answer_generator=mock_gen,
            hybrid_retriever=mock_retriever,
            reranker=MagicMock(),
            safety_guard=mock_safety,
        )
        svc._tool_registry = mock_tools
        svc._tools_checked = True
        return svc

    def test_chat_handler_records_memory(self, service):
        """Chat 模式：应当将 user 和 assistant 消息存入记忆。"""
        assert service.memory.stats["stm_count"] == 0
        result = service._handle_chat("你好", {"execution_mode": "chat"})
        assert result["answer"] == "这是一个模拟回答。"
        assert service.memory.stats["stm_count"] == 2  # user + assistant

    def test_chat_handler_injects_memory_context(self, service):
        """Chat 模式：系统提示词应包含记忆上下文。"""
        service.memory.add_message("user", "我叫张三")
        ctx_before = len(service.memory.build_context("你好"))
        with patch.object(service.answer_generator, "generate") as mock_gen:
            service._handle_chat("你好", {"execution_mode": "chat"})
            # 验证 generate 收到的系统提示词包含记忆上下文
            call_kwargs = mock_gen.call_args[0][0]
            system_msg = call_kwargs[0]["content"]
            assert "【用户偏好】" in system_msg or len(ctx_before) == 0

    def test_tool_handler_records_memory(self, service):
        """Tool 模式：应当记录用户请求和工具结果。"""
        result = service._handle_tool(
            "阿莫西林用量", "dosage_calculator", {"drug_name": "阿莫西林", "age": 30}
        )
        assert service.memory.stats["stm_count"] == 2

    def test_rag_handler_records_memory(self, service):
        """RAG 模式：应当记录用户请求和生成的回答。"""
        result = service._handle_rag("感冒了怎么办", {"execution_mode": "rag"})
        assert result["answer"] == "这是一个模拟回答。"
        assert service.memory.stats["stm_count"] >= 2

    def test_rag_handler_injects_memory_context(self, service):
        """RAG 模式：应当将记忆上下文注入用户消息（Schema-Driven Assembly）。"""
        service.memory.add_message("user", "我对青霉素过敏")
        with patch.object(service.answer_generator, "generate") as mock_gen:
            service._handle_rag("感冒了怎么办", {"execution_mode": "rag"})
            call_kwargs = mock_gen.call_args[0][0]
            user_text = call_kwargs[1]["content"]
            assert "青霉素" in user_text

    def test_react_handler_returns_answer(self, service):
        """ReAct 模式：engine 执行返回 answer 结构。"""
        with patch.object(service.answer_generator, "generate") as mock_gen:
            mock_gen.return_value = "分析结果"
            from medrag.llm.provider import get_llm_provider
            prov = get_llm_provider()
            with patch.object(prov.client.chat.completions, "create") as mock_llm:
                mock_response = type('obj', (object,), {
                    'choices': [type('obj', (object,), {'message': type('obj', (object,), {'content': '最终答案：分析结果'})})()]
                })
                mock_llm.return_value = mock_response
                result = service._handle_react("帮我分析", {"execution_mode": "react"})
                assert "answer" in result
                assert "react_trace" in result
                assert service.memory.stats["stm_count"] >= 2

    def test_stream_chat_records_memory(self, service):
        """流式 Chat：流结束后应当记录记忆。"""
        list(service._stream_chat("你好", {"execution_mode": "chat"}))
        assert service.memory.stats["stm_count"] == 2

    def test_stream_rag_records_memory(self, service):
        """流式 RAG：流结束后应当记录记忆。"""
        list(service._stream_rag("感冒了怎么办", {"execution_mode": "rag"}))
        assert service.memory.stats["stm_count"] == 2

    def test_stream_tool_records_memory(self, service):
        """流式 Tool：应当记录记忆。"""
        list(service._stream_tool("阿莫西林用量", "dosage_calculator", {}))
        assert service.memory.stats["stm_count"] == 2

    def test_stream_react_records_memory(self, service):
        """流式 ReAct：流结束后应当记录记忆。"""
        from medrag.llm.provider import get_llm_provider
        prov = get_llm_provider()
        with patch.object(prov.client.chat.completions, "create") as mock_llm:
            mock_response = type('obj', (object,), {
                'choices': [type('obj', (object,), {'message': type('obj', (object,), {'content': '最终答案：分析结果'})})()]
            })
            mock_llm.return_value = mock_response
            list(service._stream_react("帮我分析", {"execution_mode": "react"}))
            assert service.memory.stats["stm_count"] == 2

    def test_main_chat_dispatch_integration(self, service):
        """chat() 主入口：规则路由走 RAG 并记录记忆。"""
        result = service.chat("感冒了怎么办")
        assert result["answer"] == "这是一个模拟回答。"
        assert service.memory.stats["stm_count"] >= 2

    def test_main_chat_dispatch_tool_path(self, service):
        """chat() 主入口：工具匹配时走 tool 路径并记录记忆。"""
        service._tool_registry.match.return_value = ("dosage_calculator", {"drug_name": "阿莫西林"})
        service._tool_registry.execute.return_value = "成人常规剂量：0.5g qid"
        result = service.chat("阿莫西林怎么吃")
        assert service.memory.stats["stm_count"] == 2
