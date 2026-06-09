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
            if qa is not None and hasattr(qa, 'embedding_model'):
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
            return self._handle_react_stub(query, route)
        else:  # "rag"（默认）
            return self._handle_rag(query, route, user_case_summary, username)

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
            yield from self._stream_react_stub(query, route)
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
    # ReAct stub
    # ------------------------------------------------------------------

    def _handle_react_stub(self, query: str, route: dict) -> Dict:
        """ReAct stub：提示尚未启用。"""
        answer = "复杂推理模式正在开发中，请使用其他方式查询。"
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", answer)
        return {
            "answer": answer,
            "route": {**route, "execution_mode": "react"},
            "kg_results": [],
            "qa_results": [],
            "case_results": [],
            "qa_source_details": {},
            "risk_info": {"has_risk": False, "risk_keywords": []},
            "query_info": None,
        }

    def _stream_react_stub(self, query: str, route: dict) -> Generator[Dict, None, None]:
        yield {
            "type": "rag_step",
            "step": {"key": "react", "label": "复杂推理", "icon": "🧠", "detail": "模式尚未启用"},
        }
        answer = "复杂推理模式正在开发中，请使用其他方式查询。"
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", answer)
        yield {"type": "content", "content": answer}

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

        # 5. 提示词
        if user_case_summary is None and username:
            user_case_summary = get_combined_case_summary(username)
        messages = self.prompt_builder.build_messages(
            query=query,
            kg_results=retrieval["kg_results"],
            qa_results=retrieval["qa_results"],
            case_results=retrieval.get("case_results", []),
            case_context=user_case_summary,
            route=route,
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )

        # 6. 获取 query embedding（用于 LTM 语义召回增强）
        query_emb = self._get_query_embedding(query)

        # 7. 置信度联动：低置信时不注入记忆上下文（避免误导）
        if retrieval_quality.get("confidence") != "none":
            mem_context = self.memory.build_context(query, query_embedding=query_emb)
            if mem_context and messages and messages[0]["role"] == "system":
                messages[0]["content"] = f"{mem_context}\n\n{messages[0]['content']}"

        # 记录用户消息（带 embedding，提升 LTM 内联去重和召回质量）
        if query_emb is not None:
            self.memory.add_message_with_embedding("user", query, query_emb)
        else:
            self.memory.add_message("user", query)

        # 8. 生成
        answer = self.answer_generator.generate(messages)

        # 9. 安全提示
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

        # 5. 提示词
        yield {
            "type": "rag_step",
            "step": {"key": "prompt", "label": "构建提示词", "icon": "📝"},
        }
        if user_case_summary is None and username:
            user_case_summary = get_combined_case_summary(username)
        messages = self.prompt_builder.build_messages(
            query=query,
            kg_results=retrieval["kg_results"],
            qa_results=retrieval["qa_results"],
            case_results=retrieval.get("case_results", []),
            case_context=user_case_summary,
            route=route,
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )

        # 获取 query embedding（用于 LTM 语义召回增强）
        query_emb = self._get_query_embedding(query)

        # 置信度联动：低置信时不注入记忆上下文
        if retrieval_quality.get("confidence") != "none":
            mem_context = self.memory.build_context(query, query_embedding=query_emb)
            if mem_context and messages and messages[0]["role"] == "system":
                messages[0]["content"] = f"{mem_context}\n\n{messages[0]['content']}"

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
