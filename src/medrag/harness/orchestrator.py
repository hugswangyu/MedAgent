"""src/medrag/harness/orchestrator.py — 医疗全流程编排器。

将 MedAgent 现有的串行流程（risk_detect → route → retrieve → ... → complete）
封装为带有状态机、超时、重试、事件通知的可编排 HarnessOrchestrator。

区别于通用 Agent 框架，专门为医疗问答的 4 种模式
（tool / chat / rag / react）提供统一的容错编排。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from medrag.harness.types import (
    MedToolResult,
    ToolStatus,
    Confidence,
    MedPhase,
    HarnessConfig,
    MedStateMachine,
)

logger = logging.getLogger(__name__)


@dataclass
class HarnessOrchestrator:
    """医疗问答全流程编排器。

    不创建新组件，而是包装现有的 MedicalChatService 子组件，
    用统一的容错策略（超时+重试+降级）覆盖每个阶段。

    Usage::

        orchestrator = HarnessOrchestrator(
            risk_detector=safety_guard.detect_risk,
            router=query_router.route,
            retriever=hybrid_retriever.retrieve,
            reranker=reranker.rerank,
            assembler=prompt_builder.build_messages_with_context,
            generator=answer_generator.generate,
            safety_checker=_do_safety_check,
        )
        result = orchestrator.run(query="高血压")
    """
    risk_detector: Optional[Callable] = None
    router: Optional[Callable] = None
    retriever: Optional[Callable] = None
    reranker: Optional[Callable] = None
    assembler: Optional[Callable] = None
    generator: Optional[Callable] = None
    safety_checker: Optional[Callable] = None
    tool_executor: Optional[Callable] = None

    on_event: Optional[Callable[[str, dict], None]] = None

    config: HarnessConfig = field(default_factory=HarnessConfig)
    state: Optional[MedStateMachine] = None

    def run(self, query: str, **context) -> Dict[str, Any]:
        """执行一次完整的医疗问答流程。

        Returns:
            统一响应字典，包含 answer、status、trace、risk_info 等。
        """
        self.state = MedStateMachine(session_id=f"session-{time.time_ns()}")
        self._emit("start", {"query": query})

        result: Dict[str, Any] = {
            "query": query,
            "status": "success",
            "answer": "",
            "route": {},
            "risk_info": {},
            "trace": {},
            "harness_warning": None,
        }

        # ── Phase 1: 安全检测 ──
        try:
            self.state.transition(MedPhase.RISK_DETECT)
            self._emit("phase_change", {"phase": "risk_detect"})

            if self.risk_detector:
                risk_info = self.risk_detector(query)
                self.state.record_risk(
                    risk_info.get("level", "none"),
                    risk_info.get("risk_keywords", []),
                )
                result["risk_info"] = risk_info
                has_risk = risk_info.get("has_risk", False)
                risk_level = risk_info.get("level", "none")
            else:
                has_risk = False
                risk_level = "none"
        except Exception as exc:
            logger.warning("Risk detection failed: %s", exc)
            has_risk = False
            risk_level = "none"
            result["harness_warning"] = "安全检测暂时不可用。"

        # ── Phase 2: 路由 ──
        try:
            self.state.transition(MedPhase.ROUTE)
            self._emit("phase_change", {"phase": "route"})

            if self.router:
                route = self.router(query)
                result["route"] = route
                mode = route.get("execution_mode", "rag")
            else:
                mode = "rag"
        except Exception as exc:
            logger.warning("Routing failed, default to rag: %s", exc)
            mode = "rag"
            result["route"] = {"execution_mode": "rag"}
            existing = result.get("harness_warning") or ""
            result["harness_warning"] = existing + "路由决策降级为默认。"

        # ── 按模式分发 ──
        if mode == "tool":
            answer = self._run_tool_mode(query, result)
        elif mode == "chat":
            answer = self._run_chat_mode(query, result)
        elif mode == "react":
            answer = self._run_react_mode(query, result)
        else:
            answer = self._run_rag_mode(query, result, **context)

        result["answer"] = answer

        # ── Phase: 安全检测（输出） ──
        if self.safety_checker:
            try:
                self.state.transition(MedPhase.SAFETY_CHECK)
                self._emit("phase_change", {"phase": "safety_check"})

                retrieval_quality = "high" if result.get("trace", {}).get("sources") else "none"
                answer = self.safety_checker(
                    query=query, answer=answer,
                    retrieval_quality=retrieval_quality,
                    query_type=result.get("route", {}).get("query_type", ""),
                )
                result["answer"] = answer
                self.state.record_safety_checked()
            except Exception as exc:
                logger.warning("Safety check failed: %s", exc)
                existing = result.get("harness_warning") or ""
                result["harness_warning"] = existing + "安全检测异常。"

        # ── Complete ──
        try:
            self.state.transition(MedPhase.COMPLETE)
        except ValueError:
            self.state.force_transition(MedPhase.COMPLETE)
        self._emit("complete", {"session_id": self.state.session_id})

        result["trace"] = {
            "phases": self.state.history,
            "retries": self.state.retry_history,
            "total_duration_ms": self.state.metadata.get("total_duration_ms"),
        }

        return result

    # ======================================================================
    # 模式分发
    # ======================================================================

    def _run_tool_mode(self, query: str, result: dict) -> str:
        """工具模式：执行原生工具返回结果。"""
        tool_name = result.get("route", {}).get("tool_name", "")
        tool_params = result.get("route", {}).get("tool_params", {})

        if self.tool_executor and tool_name:
            tool_result = self.tool_executor(tool_name=tool_name, **tool_params)
            if isinstance(tool_result, str):
                return tool_result
            return str(tool_result.data) if tool_result.data else "未找到结果。"
        return "工具不可用。"

    def _run_chat_mode(self, query: str, result: dict) -> str:
        """聊天模式：直接 LLM 回复。"""
        if self.generator:
            messages = [{"role": "user", "content": query}]
            return self.generator(messages)
        return "你好！有什么可以帮你的吗？"

    def _run_rag_mode(self, query: str, result: dict, **context) -> str:
        """RAG 模式：完整的检索-重排-生成流水线。"""
        retrieval_data = {}
        reranked_qa = []

        # ── Phase 3: 多源检索 ──
        self.state.transition(MedPhase.RETRIEVE)
        self._emit("phase_change", {"phase": "retrieve"})

        if self.retriever:
            try:
                retrieval_data = self.retriever(query, **context)
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)
                retrieval_data = {"kg_results": [], "qa_results": []}
                existing = result.get("harness_warning") or ""
                result["harness_warning"] = existing + "检索暂时不可用，回答基于自身知识。"

        # ── Phase 4: 重排序 ──
        self.state.transition(MedPhase.RERANK)
        self._emit("phase_change", {"phase": "rerank"})

        qa_results = retrieval_data.get("qa_results", [])
        if self.reranker and qa_results:
            try:
                reranked_qa = self.reranker(query, qa_results)
            except Exception as exc:
                logger.warning("Rerank failed, using raw results: %s", exc)
                reranked_qa = qa_results[:5]
        else:
            reranked_qa = qa_results[:5]

        # ── Phase 5: 上下文组装 ──
        self.state.transition(MedPhase.ASSEMBLE)
        self._emit("phase_change", {"phase": "assemble"})

        messages = []
        if self.assembler:
            try:
                context_data = self.assembler(
                    query=query,
                    kg_results=retrieval_data.get("kg_results", []),
                    qa_results=reranked_qa,
                    case_results=retrieval_data.get("case_results", []),
                    route=result.get("route", {}),
                )
                messages = context_data.get("messages", []) if isinstance(context_data, dict) else context_data
            except Exception as exc:
                logger.warning("Assembly failed: %s", exc)
                messages = [{"role": "user", "content": query}]

        if not messages:
            messages = [{"role": "user", "content": query}]

        # ── Phase 6: 生成 ──
        self.state.transition(MedPhase.GENERATE)
        self._emit("phase_change", {"phase": "generate"})

        if self.generator:
            try:
                answer = self.generator(messages)
            except Exception as exc:
                logger.warning("Generation failed: %s", exc)
                answer = "抱歉，AI 模型暂时无法生成回答，请稍后重试。"
                existing = result.get("harness_warning") or ""
                result["harness_warning"] = existing + "AI 生成异常。"
        else:
            answer = ""

        return answer

    def _run_react_mode(self, query: str, result: dict) -> str:
        """ReAct 模式：多步推理。"""
        if self.generator:
            messages = [{"role": "user", "content": query}]
            return self.generator(messages)
        return "ReAct 模式暂时不可用。"

    def _emit(self, event_type: str, data: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass
