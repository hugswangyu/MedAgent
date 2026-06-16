"""src/medrag/harness/orchestrator.py — 医疗全流程编排器（ReAct 统一架构）。

所有查询统一进入 ReAct 循环，LLM 自主决定调工具还是直接回答。
RAG 检索作为 ``retrieve_knowledge`` 工具在循环内调用。

流程::

    risk_detect → route(仅 metadata) → react_loop → safety_check → complete
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from medrag.harness.types import (
    MedPhase,
    HarnessConfig,
    MedStateMachine,
)

logger = logging.getLogger(__name__)


@dataclass
class HarnessOrchestrator:
    """医疗问答全流程编排器（ReAct 统一架构）。

    不再按 mode 分发到不同执行路径，所有查询统一走 ReAct 循环。
    旧的 retriever/reranker/assembler/generator 字段保留但不再使用，
    需通过 ``react_engine_builder`` 提供 ReAct 引擎的构建回调。

    Usage::

        orchestrator = HarnessOrchestrator(
            risk_detector=safety_guard.detect_risk,
            router=query_router.route,
            react_engine_builder=_build_react_engine,
            safety_checker=_do_safety_check,
        )
        result = orchestrator.run(query="高血压")
    """
    risk_detector: Optional[Callable] = None
    router: Optional[Callable] = None

    # ── ReAct 引擎构建器 ──
    # 签名: react_engine_builder(query, route) -> ReActEngine
    # 由 chat_service 负责创建并注册所有工具
    react_engine_builder: Optional[Callable] = None

    # ── 废弃字段（保留以兼容旧调用方，不再使用） ──
    retriever: Optional[Callable] = None     # deprecated
    reranker: Optional[Callable] = None      # deprecated
    assembler: Optional[Callable] = None     # deprecated
    generator: Optional[Callable] = None     # deprecated
    tool_executor: Optional[Callable] = None  # deprecated

    safety_checker: Optional[Callable] = None
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
            "react_trace": None,
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
            else:
                risk_info = {"has_risk": False, "level": "none"}
                result["risk_info"] = risk_info
        except Exception as exc:
            logger.warning("Risk detection failed: %s", exc)
            risk_info = {"has_risk": False, "level": "none"}
            result["risk_info"] = risk_info
            result["harness_warning"] = "安全检测暂时不可用。"

        # ── Phase 2: 路由（仅元数据，不选执行模式） ──
        try:
            self.state.transition(MedPhase.ROUTE)
            self._emit("phase_change", {"phase": "route"})

            if self.router:
                route = self.router(query)
                result["route"] = route
            else:
                result["route"] = {}
        except Exception as exc:
            logger.warning("Route failed: %s", exc)
            result["route"] = {}
            existing = result.get("harness_warning") or ""
            result["harness_warning"] = existing + "路由降级。"

        # ── Phase 3: ReAct 循环 ──
        self.state.transition(MedPhase.REACT_LOOP)
        self._emit("phase_change", {"phase": "react_loop"})

        if self.react_engine_builder:
            try:
                engine = self.react_engine_builder(query, result.get("route", {}))
                engine_result = engine.run(query, system_context="")
                result["answer"] = engine_result.get("answer", "")
                result["react_trace"] = {
                    "steps": engine_result.get("steps", []),
                    "tool_results": engine_result.get("tool_results", {}),
                }
            except Exception as exc:
                logger.error("ReAct loop failed: %s", exc)
                result["status"] = "failed"
                result["answer"] = "抱歉，AI 暂时无法回答，请稍后重试。"
                existing = result.get("harness_warning") or ""
                result["harness_warning"] = existing + "推理异常。"
        else:
            result["answer"] = "系统尚未配置推理引擎。"

        # ── Phase 4: 安全检测（输出） ──
        if self.safety_checker and result.get("answer"):
            try:
                self.state.transition(MedPhase.SAFETY_CHECK)
                self._emit("phase_change", {"phase": "safety_check"})

                retrieval_quality = (
                    "high" if result.get("react_trace", {}).get("steps") else "none"
                )
                answer = self.safety_checker(
                    query=query,
                    answer=result["answer"],
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

    def _emit(self, event_type: str, data: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass
