"""统一医疗对话服务：完整 RAG 流水线的单一入口。

编排：检索 → 重排序 → 提示词构建 → 生成 → 安全检查。
"""

from __future__ import annotations

import logging
from typing import Dict, Generator, Optional

import numpy as np

from medrag.config.settings import settings
from medrag.llm import get_llm_client, get_llm_provider
from medrag.rag import PromptBuilder, AnswerGenerator, SafetyGuard
from medrag.retrieval import (
    HybridRetriever,
    KGRetriever,
    QueryNormalizer,
    QueryRouter,
    get_reranker,
)
from medrag.memory import MemorySystem, get_memory_system
from medrag.data.user_case_store import UserCaseRetriever, get_combined_case_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MedicalChatService
# ---------------------------------------------------------------------------


class MedicalChatService:
    """端到端医疗问答流水线。

    用法::

        service = MedicalChatService()
        result = service.chat("感冒了怎么办")

    所有子组件均可注入以便测试或自定义配置::

        service = MedicalChatService(
            kg_retriever=my_kg,
            answer_generator=my_gen,
        )
    """

    def __init__(
        self,
        kg_retriever=None,          # KGRetriever 实例 或 None → 自动加载
        qa_retriever=None,          # QARetriever 或 None → 自动创建
        es_retriever=None,          # ESBM25Retriever 或 None → 自动创建
        router=None,                # QueryRouter 或 None → 自动创建
        hybrid_retriever=None,      # HybridRetriever 或 None → 由上述组件组装
        reranker=None,              # reranker 实例 或 None → get_reranker()
        prompt_builder=None,        # PromptBuilder 或 None → 自动创建
        answer_generator=None,      # AnswerGenerator 或 None → 自动创建
        safety_guard=None,          # SafetyGuard 或 None → 自动创建
        case_retriever=None,        # UserCaseRetriever 或 None → 自动创建
        normalizer=None,            # QueryNormalizer 或 None → 自动创建
        memory_system=None,         # MemorySystem 或 None → 自动创建单例
        memory_persist_path=None,   # Memory JSON 持久化路径（None=不持久化）
    ):
        # ---- 共享 LLM 客户端 ----
        _llm_client = get_llm_client()
        _llm_provider = get_llm_provider()

        # ---- 检索流水线 ----
        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        else:
            if kg_retriever is None:
                from pathlib import Path
                from medrag.infrastructure.ner import load_ner_model
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                kg_retriever = load_ner_model(project_root, llm_client=_llm_client)

            from medrag.vectors.qa_retriever import QARetriever
            _qa = qa_retriever or QARetriever()

            from medrag.retrieval.es_retriever import ESBM25Retriever
            _es = es_retriever or ESBM25Retriever()

            _router = router or QueryRouter(llm_client=_llm_client)

            self.hybrid_retriever = HybridRetriever(
                kg_retriever=kg_retriever,
                qa_retriever=_qa,
                es_retriever=_es,
                router=_router,
                case_retriever=case_retriever or UserCaseRetriever(),
                normalizer=normalizer or QueryNormalizer(),
            )

            # ---- 组件健康检查 ----
            from medrag.infrastructure.health import report_ok, report_down
            try:
                if _es.client.ping():
                    report_ok("elasticsearch")
                else:
                    report_down("elasticsearch", "ES ping returned False")
            except Exception as exc:
                report_down("elasticsearch", str(exc))

            if kg_retriever is not None:
                try:
                    kg_retriever.neo4j.run("RETURN 1")
                    report_ok("neo4j")
                except Exception as exc:
                    report_down("neo4j", str(exc))

            if hasattr(_qa, '_available'):
                report_ok("milvus") if _qa._available else report_down("milvus", "QA retriever unavailable")

        # ---- 生成流水线 ----
        self.reranker = reranker or get_reranker()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.answer_generator = answer_generator or AnswerGenerator(llm_provider=_llm_provider)
        self.safety_guard = safety_guard or SafetyGuard()

        # ---- 记忆系统 ----
        if memory_system is not None:
            self.memory = memory_system
        elif memory_persist_path is not None:
            from medrag.memory import create_memory_system
            self.memory = create_memory_system(persist_path=memory_persist_path)
        else:
            self.memory = get_memory_system()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    # 延迟导入，避免循环依赖
    _tools_checked = False

    def _get_tool_registry(self):
        if not self._tools_checked:
            from medrag.tools import get_tool_registry
            self._tool_registry = get_tool_registry()
            type(self)._tools_checked = True
        return self._tool_registry

    def _get_query_embedding(self, query: str):
        """Try to extract query embedding from the QA retriever for memory storage.

        Returns np.ndarray or None if embedding model is unavailable.
        """
        try:
            qa = getattr(self.hybrid_retriever, 'qa', None)
            if qa is not None and hasattr(qa, 'embedding_model') and qa.embedding_model is not None:
                return np.array(qa.embedding_model.encode_one(query, is_query=True))
        except Exception:
            logger.debug("Query embedding unavailable for memory", exc_info=True)
        return None

    def chat(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict:
        """运行多模式医疗问答。

        自动检查三个内置工具（剂量计算/科室导诊/正常值查询），
        不命中则按 Router 分发的执行模式处理。
        """
        # 0. 优先检查工具匹配（快速路径，不走 Router/Retrieval）
        tool_name, tool_params = self._get_tool_registry().match(query)
        if tool_name is not None:
            return self._handle_tool(query, tool_name, tool_params)

        # 1. 路由 — 获取 execution_mode
        route = self.hybrid_retriever.router.route(query)
        mode = route.get("execution_mode", "rag")

        # 2. 按模式分发
        if mode == "chat":
            return self._handle_chat(query, route)
        elif mode == "react":
            return self._handle_react(query, route)
        else:  # "rag"（默认）
            return self._handle_rag(query, route, user_case_summary, username)

    def chat_with_harness(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict:
        """Harness 版本的 chat() — 统一容错编排。

        用法与 chat() 完全一致，通过 HarnessOrchestrator
        对每个阶段进行超时 + 重试 + 中文降级提示。
        """
        from medrag.harness.orchestrator import HarnessOrchestrator

        # 0. 工具快速路径（与 chat() 保持一致）
        tool_name, tool_params = self._get_tool_registry().match(query)
        if tool_name is not None:
            return self._handle_tool(query, tool_name, tool_params)

        # 1. 组装 orchestrator 回调
        def _risk_detect(query):
            return self.safety_guard.detect_risk(query)

        def _route(query):
            return self.hybrid_retriever.router.route(query)

        def _retrieve(query, username=None):
            return self.hybrid_retriever.retrieve(query, username=username)

        def _rerank(query, results):
            return self.reranker.rerank(query, results, top_k=settings.rerank_top_k)

        def _assemble(**kw):
            query = kw.get("query", "")
            kg_results = kw.get("kg_results", [])
            qa_results = kw.get("qa_results", [])
            route = kw.get("route", {})
            retrieval_quality = {
                "has_kg": bool(kg_results),
                "has_qa": bool(qa_results),
                "confidence": "high" if (kg_results or qa_results) else "none",
            }
            sections = self.prompt_builder.build_sections(
                query=query,
                kg_results=kg_results,
                qa_results=qa_results,
                route=route,
                retrieval_quality=retrieval_quality,
            )

            from medrag.memory import ContextAssembler
            assembler = ContextAssembler(budget=getattr(settings, 'context_budget', 3000))
            from medrag.memory.schema import (
                PRIORITY_KG, PRIORITY_QA,
            )
            for name, text in sections.items():
                priority = {"kg": PRIORITY_KG, "qa": PRIORITY_QA}.get(name, 50)
                assembler.add(name, text, priority=priority)

            context = assembler.assemble()
            messages = self.prompt_builder.build_messages_with_context(
                context=context,
                query=query,
                route=route,
                retrieval_quality=retrieval_quality,
            )
            return {"messages": messages}

        def _generate(messages):
            return self.answer_generator.generate(messages)

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

        # 2. 记录用户消息
        query_emb = self._get_query_embedding(query)
        if query_emb is not None:
            self.memory.add_message_with_embedding("user", query, query_emb)
        else:
            self.memory.add_message("user", query)

        # 3. 执行编排
        result = orchestrator.run(query=query, username=username)

        # 4. 记录助手回复
        self.memory.store_assistant_reply(result.get("answer", ""))

        # 5. 兼容现有返回格式
        return {
            "answer": result["answer"],
            "route": result.get("route", {}),
            "risk_info": result.get("risk_info", {}),
            "harness_trace": result.get("trace", {}),
            "harness_warning": result.get("harness_warning"),
            "kg_results": [],
            "qa_results": [],
            "case_results": [],
            "qa_source_details": {},
            "query_info": None,
        }

    def stream_chat(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
        department: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """多模式流式问答。

        模式自适应：工具 → Chat → RAG → ReAct(stub)，自动选择。
        """
        # 0. 优先检查工具匹配
        tool_name, tool_params = self._get_tool_registry().match(query)
        if tool_name is not None:
            yield from self._stream_tool(query, tool_name, tool_params)
            return

        # 1. 路由
        route = self.hybrid_retriever.router.route(query)
        mode = route.get("execution_mode", "rag")

        # 2. 分发
        if mode == "chat":
            yield from self._stream_chat(query, route, provider, model)
        elif mode == "react":
            yield from self._stream_react(query, route)
        else:
            yield from self._stream_rag(
                query, route, user_case_summary, username,
                department, provider, model,
            )

    # ------------------------------------------------------------------
    # Chat 模式
    # ------------------------------------------------------------------

    def _handle_chat(self, query: str, route: dict) -> Dict:
        """Chat 模式：直接 LLM 回复，不走 RAG。"""
        # 记忆上下文构建
        mem_context = self.memory.build_context(query)
        system_content = "你是一个友好的医疗助手。请以亲切、自然的语气简短回复用户的问候或闲聊。"
        if mem_context:
            system_content = f"{mem_context}\n\n{system_content}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        answer = self.answer_generator.generate(messages)

        # 记录到记忆系统
        self.memory.add_message("user", query)
        self.memory.store_assistant_reply(answer)

        return {
            "answer": answer,
            "route": route,
            "kg_results": [],
            "qa_results": [],
            "case_results": [],
            "qa_source_details": {},
            "risk_info": {"has_risk": False, "risk_keywords": []},
            "query_info": None,
        }

    def _stream_chat(
        self, query: str, route: dict,
        provider: Optional[str] = None, model: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """流式 Chat 模式。"""
        if provider and provider != settings.llm_provider:
            _gen_provider = get_llm_provider(provider)
            answer_gen = AnswerGenerator(llm_provider=_gen_provider)
        else:
            answer_gen = self.answer_generator

        yield {
            "type": "rag_step",
            "step": {"key": "chat", "label": "闲聊回复", "icon": "💬"},
        }
        # 记忆上下文构建
        mem_context = self.memory.build_context(query)
        system_content = "你是一个友好的医疗助手。请以亲切、自然的语气简短回复用户的问候或闲聊。"
        if mem_context:
            system_content = f"{mem_context}\n\n{system_content}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        # 记录用户消息
        self.memory.add_message("user", query)

        full_answer = ""
        for token in answer_gen.generate_stream(messages, model=model):
            full_answer += token
            yield {"type": "content", "content": token}

        # 记录助手回复
        self.memory.store_assistant_reply(full_answer)

    # ------------------------------------------------------------------
    # Tool 模式
    # ------------------------------------------------------------------

    def _handle_tool(self, query: str, tool_name: str, tool_params: dict) -> Dict:
        """Tool 模式：执行原生工具，直接返回结构化结果。"""
        # 记录用户消息
        self.memory.add_message("user", query)

        result = self._tool_registry.execute(tool_name, **tool_params)

        # 记录助手回复
        self.memory.add_message("assistant", result)

        return {
            "answer": result,
            "route": {"execution_mode": "tool", "tool_name": tool_name},
            "kg_results": [],
            "qa_results": [],
            "case_results": [],
            "qa_source_details": {},
            "risk_info": {"has_risk": False, "risk_keywords": []},
            "query_info": None,
        }

    def _stream_tool(
        self, query: str, tool_name: str, tool_params: dict,
    ) -> Generator[Dict, None, None]:
        """流式 Tool 模式：单次 yield 完整结果。"""
        yield {
            "type": "rag_step",
            "step": {"key": "tool", "label": "工具调用", "icon": "🔧", "detail": tool_name},
        }
        self.memory.add_message("user", query)
        result = self._tool_registry.execute(tool_name, **tool_params)
        self.memory.add_message("assistant", result)
        yield {"type": "content", "content": result}

    # ------------------------------------------------------------------
    # ReAct 多步推理
    # ------------------------------------------------------------------

    def _handle_react(self, query: str, route: dict) -> Dict:
        """ReAct 模式：多步推理循环（Thought/Action/Observation）。"""
        risk_info = self.safety_guard.detect_risk(query)

        provider = get_llm_provider()
        from medrag.react import ReActEngine

        engine = ReActEngine(provider.client, model=provider.default_model)

        # ── 注册检索工具 ──
        kg = getattr(self.hybrid_retriever, 'kg', None)
        if kg is not None:
            engine.register_tool(
                "retrieve_kg", "查询医学知识图谱，获取疾病、症状、药物间的结构化关系",
                executor=kg.search,
                parameters=[{"name": "query", "description": "搜索关键词", "type": "string"}],
            )
        engine.register_tool(
            "retrieve_qa", "检索医疗问答库和知识图谱，查找类似病例和医学知识",
            executor=self._react_qa_search,
            parameters=[{"name": "query", "description": "搜索关键词", "type": "string"}],
        )

        # ── 注册计算工具 ──
        from medrag.tools.dosage_calculator import DosageCalculator
        _dosage = DosageCalculator()
        engine.register_tool(
            "calculate_dosage", "计算常见药物剂量，支持成人和儿童剂量换算",
            executor=lambda **kw: _dosage.execute(**kw),
            parameters=[
                {"name": "drug_name", "description": "药物名称（中文）", "type": "string"},
                {"name": "age", "description": "患者年龄（岁）", "type": "number"},
                {"name": "weight", "description": "患者体重（kg，可选）", "type": "number"},
            ],
        )
        from medrag.tools.normal_range import NormalRangeTool
        _normal = NormalRangeTool()
        engine.register_tool(
            "query_normal_range", "查询医学检查项目的正常值参考范围，可判断检查值是否正常",
            executor=lambda **kw: _normal.execute(**kw),
            parameters=[
                {"name": "test_name", "description": "检查项目名称（中文）", "type": "string"},
                {"name": "value", "description": "检查值（可选，传入则判断是否正常）", "type": "string"},
            ],
        )

        # ── 记忆上下文 ──
        query_emb = self._get_query_embedding(query)
        mem_context = self.memory.build_context(query, query_embedding=query_emb)

        # ── 执行 ReAct 循环 ──
        result = engine.run(query, system_context=mem_context)

        # ── 记录记忆 ──
        if query_emb is not None:
            self.memory.add_message_with_embedding("user", query, query_emb)
        else:
            self.memory.add_message("user", query)
        self.memory.store_assistant_reply(result["answer"])

        return {
            "answer": result["answer"],
            "route": {**route, "execution_mode": "react"},
            "kg_results": [],
            "qa_results": [],
            "case_results": [],
            "qa_source_details": {},
            "risk_info": risk_info,
            "query_info": None,
            "react_trace": {
                "steps": result.get("steps", []),
                "tool_results": result.get("tool_results", {}),
            },
        }

    def _react_qa_search(self, query: str) -> str:
        """ReAct QA/知识库检索辅助。"""
        try:
            retrieval = self.hybrid_retriever.retrieve(query)
            kg = retrieval.get("kg_results", [])
            qa = retrieval.get("qa_results", [])
            parts = []
            if kg:
                parts.append("【知识图谱结果】")
                for r in kg[:3]:
                    parts.append(f"- {r.get('answer', '')[:400]}")
            if qa:
                parts.append("【相似问答结果】")
                for r in qa[:3]:
                    ans = r.get('answer') or r.get('text', '')
                    parts.append(f"- {ans[:400]}")
            return "\n".join(parts) if parts else "未找到相关信息"
        except Exception as exc:
            logger.warning("ReAct QA search failed: %s", exc)
            return f"检索出错：{exc}"

    def _stream_react(self, query: str, route: dict) -> Generator[Dict, None, None]:
        """流式 ReAct 模式：逐步产出推理步骤，最后流式输出答案。"""
        yield {
            "type": "rag_step",
            "step": {"key": "react", "label": "ReAct 推理", "icon": "🧠", "detail": "多步推理"},
        }
        result = self._handle_react(query, route)

        # 产出推理轨迹
        for step in result.get("react_trace", {}).get("steps", []):
            yield {
                "type": "rag_step",
                "step": {
                    "key": f"react_step_{step['step']}",
                    "label": f"步骤 {step['step']}",
                    "icon": "🔍",
                    "detail": f"{step['action']}: {step['thought'][:80]}",
                },
            }

        yield {"type": "content", "content": result["answer"]}

    # ------------------------------------------------------------------
    # RAG 模式（原有完整流水线）
    # ------------------------------------------------------------------

    def _handle_rag(
        self,
        query: str,
        route: dict,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict:
        """RAG 模式：完整的检索-重排序-生成流水线。"""
        # 1. 风险检测
        risk_info = self.safety_guard.detect_risk(query)

        # 2. 多源检索
        retrieval = self.hybrid_retriever.retrieve(query, username=username)
        retrieval["route"] = route  # 保留外部 execution_mode

        # 3. 重排序
        retrieval["qa_results"] = self.reranker.rerank(
            query, retrieval["qa_results"], top_k=settings.rerank_top_k,
        )

        # 4. 检索质量
        retrieval_quality = {
            "has_kg": bool(retrieval["kg_results"]),
            "has_qa": bool(retrieval["qa_results"]),
            "has_case": bool(retrieval.get("case_results")),
            "confidence": (
                "high" if (retrieval["kg_results"] or retrieval["qa_results"] or retrieval.get("case_results"))
                else "none"
            ),
        }

        # 5. 获取 query embedding（用于 LTM 语义召回增强）
        query_emb = self._get_query_embedding(query)

        # 6. Schema-Driven Context Assembly（优先级 + Token 预算）
        from medrag.memory import ContextAssembler
        from medrag.memory.schema import (
            PRIORITY_MEMORY, PRIORITY_CASE_SUMMARY, PRIORITY_KG,
            PRIORITY_QA, PRIORITY_CASE_CHUNKS,
        )

        assembler = ContextAssembler(budget=settings.context_budget)

        # 6a. 记忆上下文（最高优先级）
        if retrieval_quality.get("confidence") != "none":
            mem_context = self.memory.build_context(query, query_embedding=query_emb)
            assembler.add("memory", mem_context, priority=PRIORITY_MEMORY)

        # 6b. 检索结果 sections
        if user_case_summary is None and username:
            user_case_summary = get_combined_case_summary(username)
        sections = self.prompt_builder.build_sections(
            query=query,
            kg_results=retrieval["kg_results"],
            qa_results=retrieval["qa_results"],
            case_results=retrieval.get("case_results", []),
            case_context=user_case_summary,
            route=route,
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )

        # Map section name → priority (higher = kept first under budget)
        section_priorities = {
            "case_summary": PRIORITY_CASE_SUMMARY,
            "case_chunks": PRIORITY_CASE_CHUNKS,
            "kg": PRIORITY_KG,
            "qa": PRIORITY_QA,
            "query": 50,
        }
        for name, text in sections.items():
            assembler.add(name, text, priority=section_priorities.get(name, 50))

        # 6c. Assemble with budget pruning
        context = assembler.assemble()
        messages = self.prompt_builder.build_messages_with_context(
            context=context,
            query=query,
            route=route,
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )

        # 记录用户消息（带 embedding，提升 LTM 内联去重和召回质量）
        if query_emb is not None:
            self.memory.add_message_with_embedding("user", query, query_emb)
        else:
            self.memory.add_message("user", query)

        # 7. 生成
        answer = self.answer_generator.generate(messages)

        # 8. 安全提示
        answer = self.safety_guard.append_safety_notice(
            answer, risk_info,
            retrieval_quality=retrieval_quality["confidence"],
            query_type=route.get("query_type"),
        )

        # 记录助手回复
        self.memory.store_assistant_reply(answer)

        return {
            "answer": answer,
            "route": route,
            "kg_results": retrieval["kg_results"],
            "qa_results": retrieval["qa_results"],
            "case_results": retrieval.get("case_results", []),
            "qa_source_details": retrieval.get("qa_source_details", {}),
            "risk_info": risk_info,
            "query_info": retrieval.get("query_info"),
        }

    def _stream_rag(
        self,
        query: str,
        route: dict,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
        department: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """流式 RAG 模式。"""
        # 动态创建 AnswerGenerator
        if provider and provider != settings.llm_provider:
            _gen_provider = get_llm_provider(provider)
            answer_gen = AnswerGenerator(llm_provider=_gen_provider)
        else:
            answer_gen = self.answer_generator

        # 1. 风险检测
        yield {
            "type": "rag_step",
            "step": {"key": "risk", "label": "安全检测", "icon": "🛡️", "detail": "扫描风险关键词"},
        }
        risk_info = self.safety_guard.detect_risk(query)

        # 2. 多源检索
        yield {
            "type": "rag_step",
            "step": {"key": "retrieve", "label": "多源检索", "icon": "🔍", "detail": "知识图谱 + 向量库"},
        }
        retrieval = self.hybrid_retriever.retrieve(query, department=department, username=username)
        retrieval["route"] = route

        # 3. 重排序
        yield {
            "type": "rag_step",
            "step": {"key": "rerank", "label": "QA结果重排序", "icon": "📊", "detail": f"共 {len(retrieval['qa_results'])} 条候选"},
        }
        retrieval["qa_results"] = self.reranker.rerank(
            query, retrieval["qa_results"], top_k=settings.rerank_top_k,
        )

        # 4. 检索质量
        retrieval_quality = {
            "has_kg": bool(retrieval["kg_results"]),
            "has_qa": bool(retrieval["qa_results"]),
            "has_case": bool(retrieval.get("case_results")),
            "confidence": (
                "high" if (retrieval["kg_results"] or retrieval["qa_results"] or retrieval.get("case_results"))
                else "none"
            ),
        }

        # 5. 提示词（Schema-Driven Context Assembly）
        yield {
            "type": "rag_step",
            "step": {"key": "prompt", "label": "构建提示词", "icon": "📝"},
        }
        if user_case_summary is None and username:
            user_case_summary = get_combined_case_summary(username)

        # 获取 query embedding（用于 LTM 语义召回增强）
        query_emb = self._get_query_embedding(query)

        # Schema-Driven Context Assembly
        from medrag.memory import ContextAssembler
        from medrag.memory.schema import (
            PRIORITY_MEMORY, PRIORITY_CASE_SUMMARY, PRIORITY_KG,
            PRIORITY_QA, PRIORITY_CASE_CHUNKS,
        )

        assembler = ContextAssembler(budget=settings.context_budget)

        # 记忆上下文（最高优先级）
        if retrieval_quality.get("confidence") != "none":
            mem_context = self.memory.build_context(query, query_embedding=query_emb)
            assembler.add("memory", mem_context, priority=PRIORITY_MEMORY)

        # 检索结果 sections
        sections = self.prompt_builder.build_sections(
            query=query,
            kg_results=retrieval["kg_results"],
            qa_results=retrieval["qa_results"],
            case_results=retrieval.get("case_results", []),
            case_context=user_case_summary,
            route=route,
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )
        section_priorities = {
            "case_summary": PRIORITY_CASE_SUMMARY,
            "case_chunks": PRIORITY_CASE_CHUNKS,
            "kg": PRIORITY_KG,
            "qa": PRIORITY_QA,
            "query": 50,
        }
        for name, text in sections.items():
            assembler.add(name, text, priority=section_priorities.get(name, 50))

        context = assembler.assemble()
        messages = self.prompt_builder.build_messages_with_context(
            context=context,
            query=query,
            route=route,
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )

        # 记录用户消息（带 embedding，提升 LTM 内联去重和召回质量）
        if query_emb is not None:
            self.memory.add_message_with_embedding("user", query, query_emb)
        else:
            self.memory.add_message("user", query)

        # 溯源信息
        rag_trace = {
            "tool_used": True,
            "tool_name": "multi-source-retrieval",
            "retrieval_stage": "initial",
            "retrieval_mode": route.get("query_type", ""),
            "query_info": retrieval.get("query_info"),
            "initial_retrieved_chunks": [
                {
                    "filename": r.get("source", r.get("id", "")),
                    "text": (r.get("answer") or r.get("text") or r.get("evidence", ""))[:200],
                    "rrf_rank": i + 1,
                    "rrf_score": r.get("rrf_score", 0),
                    "rerank_score": r.get("ce_score") or r.get("final_score", 0),
                    "source_rank": r.get("rrf_source_rank", 0),
                }
                for i, r in enumerate(
                    (retrieval["kg_results"] + retrieval["qa_results"])[:10]
                )
            ],
        }
        yield {"type": "trace", "rag_trace": rag_trace}

        # 6. 流式生成
        yield {
            "type": "rag_step",
            "step": {"key": "generate", "label": "生成回答", "icon": "✨"},
        }
        full_answer = ""
        for token in answer_gen.generate_stream(messages, model=model):
            full_answer += token
            yield {"type": "content", "content": token}

        # 7. 安全提示尾注
        footer = self.safety_guard.append_safety_notice(
            "", risk_info,
            retrieval_quality=retrieval_quality["confidence"],
            query_type=route.get("query_type"),
        )
        if footer.strip():
            yield {"type": "content", "content": "\n\n" + footer}

        # 记录助手回复
        self.memory.store_assistant_reply(full_answer + ("\n\n" + footer if footer.strip() else ""))
