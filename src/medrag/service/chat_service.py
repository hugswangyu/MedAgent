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
    QueryNormalizer,
    QueryRouter,
    get_reranker,
)
from medrag.memory import MemorySystem, get_memory_system
from medrag.data.user_case_store import UserCaseRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MedicalChatService
# ---------------------------------------------------------------------------


class MedicalChatService:
    """端到端医疗问答流水线。

    用法::

        service = MedicalChatService()
        result = service.chat_with_harness("感冒了怎么办")

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


    def chat_with_harness(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict:
        """Harness 版本的 chat() — 统一 ReAct 编排。

        用法与 chat() 完全一致，所有查询统一走 ReAct 循环，
        RAG 检索作为 ``retrieve_knowledge`` 工具在循环内由 LLM 自主调用。
        """
        from medrag.harness.orchestrator import HarnessOrchestrator

        # 0. 工具快速路径
        tool_name, tool_params = self._get_tool_registry().match(query)
        if tool_name is not None:
            return self._handle_tool(query, tool_name, tool_params)

        # 清除前一次查询的跟踪缓存
        self.hybrid_retriever._last_raw_result = None
        self.hybrid_retriever._last_reranked_qa = None

        # 1. 路由（仅需一次，后续所有组件共享此结果）
        route = self.hybrid_retriever.router.route(query)

        # 将路由结果注入 retriever，使其内部不再重复调用 router.route()
        self.hybrid_retriever._current_route = route

        # 2. 记录用户消息
        query_emb = self._get_query_embedding(query)
        if query_emb is not None:
            self.memory.add_message_with_embedding("user", query, query_emb)
        else:
            self.memory.add_message("user", query)

        # 3. 所有查询统一走 ReAct 编排
        def _risk_detect(query):
            return self.safety_guard.detect_risk(query)

        def _route(query):
            # 返回已确定的路由，避免 orchestrator 内部重复调用
            return route

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
            react_engine_builder=self._build_react_engine,
            safety_checker=_safety_check,
        )

        result = orchestrator.run(query=query, username=username)

        # 5. 记录助手回复
        self.memory.store_assistant_reply(result.get("answer", ""))

        # 清除路由缓存，防止泄漏到下一次查询
        self.hybrid_retriever._current_route = None

        return {
            "answer": result["answer"],
            "route": route,
            "risk_info": result.get("risk_info", {}),
            "harness_trace": result.get("trace", {}),
            "harness_warning": result.get("harness_warning"),
            "react_trace": result.get("react_trace"),
            "kg_results": [],
            "qa_results": [],
            "case_results": [],
            "qa_source_details": {},
            "query_info": None,
        }

    def _build_react_engine(self, query: str, route: dict):
        """构建注入到 ReAct 循环的引擎，注册所有可用工具。"""
        provider = get_llm_provider()
        from medrag.react import ReActEngine
        from medrag.react.rag_tool import RetrieveKnowledgeTool
        from medrag.react.tools import base_tool_to_react_tool
        from medrag.tools.dosage_calculator import DosageCalculator
        from medrag.tools.department_guide import DepartmentGuide
        from medrag.tools.normal_range import NormalRangeTool

        engine = ReActEngine(
            provider.client,
            model=provider.default_model,
            request_timeout=60.0,  # ReAct 多步，给 LLM 更长时间
        )

        # ── 注册 RAG 检索工具 ──
        knowledge_tool = RetrieveKnowledgeTool(
            self.hybrid_retriever,
            self.reranker,
            prompt_builder=self.prompt_builder,
        )
        engine.register_tool(
            knowledge_tool.name,
            knowledge_tool.description,
            executor=knowledge_tool.execute,
            parameters=knowledge_tool.parameters,
        )

        # ── 注册原生工具 ──
        for tool in [DosageCalculator(), DepartmentGuide(), NormalRangeTool()]:
            rt = base_tool_to_react_tool(tool)
            engine.register_tool_from_def(rt)

        return engine

    def stream_chat(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
        department: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """流式医疗问答 — 统一走 Harness + ReAct 路径。

        provider/model/department 参数保留以兼容外部调用方，由默认配置控制。
        """
        # 0. 优先检查工具匹配（快速路径）
        tool_name, tool_params = self._get_tool_registry().match(query)
        if tool_name is not None:
            yield from self._stream_tool(query, tool_name, tool_params)
            return

        # 1. 进度事件
        yield {
            "type": "rag_step",
            "step": {"key": "risk", "label": "安全检测", "icon": "🛡️", "detail": "扫描风险关键词"},
        }
        yield {
            "type": "rag_step",
            "step": {"key": "retrieve", "label": "多源检索", "icon": "🔍", "detail": "知识图谱 + 向量库"},
        }

        # 2. 执行 Harness 编排
        result = self.chat_with_harness(
            query, user_case_summary=user_case_summary, username=username,
        )

        # 3. 溯源信息（前端展示来源和相关度）
        # 从 ReAct/Fast 路径的缓存中读取，避免重复检索
        raw = getattr(self.hybrid_retriever, '_last_raw_result', None)
        reranked = getattr(self.hybrid_retriever, '_last_reranked_qa', None)
        if raw is not None and reranked is not None:
            try:
                kg_results = raw.get("kg_results", [])
                qa_results = reranked
                chunks = []
                logger.debug(
                    "TRACE kg_results=%d qa_results=%d",
                    len(kg_results), len(qa_results),
                )
                for i, r in enumerate((kg_results + qa_results)[:10]):
                    score = r.get("rrf_score") or r.get("score") or 0
                    rerank = r.get("ce_score") or r.get("final_score") or score
                    logger.debug(
                        "TRACE chunk[%d] source=%s score=%s rerank=%s",
                        i, r.get("source"), score, rerank,
                    )
                    chunks.append({
                        "filename": r.get("source", r.get("id", "")),
                        "text": (r.get("answer") or r.get("text") or "")[:200],
                        "rrf_rank": i + 1,
                        "rrf_score": score,
                        "rerank_score": rerank,
                        "source_rank": r.get("rrf_source_rank", 0),
                    })
                yield {"type": "trace", "rag_trace": {
                    "tool_used": True,
                    "tool_name": "multi-source-retrieval",
                    "retrieval_stage": "initial",
                    "initial_retrieved_chunks": chunks,
                }}
            except Exception:
                logger.debug("Trace assembly failed", exc_info=True)

        # 4. 产出推理轨迹
        react_trace = result.get("react_trace") or {}
        if react_trace.get("steps"):
            yield {
                "type": "rag_step",
                "step": {"key": "react", "label": "ReAct 推理", "icon": "🧠", "detail": "多步推理"},
            }
            for step in react_trace["steps"]:
                yield {
                    "type": "rag_step",
                    "step": {
                        "key": f"react_step_{step['step']}",
                        "label": f"步骤 {step['step']}",
                        "icon": "🔍",
                        "detail": f"{step['action']}: {step.get('thought', '')[:80]}",
                    },
                }

        # 5. 产出回答
        yield {"type": "content", "content": result.get("answer", "")}

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


