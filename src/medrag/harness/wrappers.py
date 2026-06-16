"""src/medrag/harness/wrappers.py — 医疗工具包装器。

将现有 MedAgent 组件（HybridRetriever、AnswerGenerator、SafetyGuard、BaseTool）
包装为统一的 ``MedToolWrapper``，每个包装器负责：
  1. 执行 + 超时 + 重试
  2. 异常 → MedToolResult 转换（含面向用户的降级提示）
  3. 证据溯源（SourceTrace）和置信度计算
  4. 重试事件记录
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from medrag.harness.types import (
    MedToolResult,
    ToolStatus,
    Confidence,
    SourceTrace,
    HarnessConfig,
)

logger = logging.getLogger(__name__)


# =========================================================================
# MedToolWrapper — 统一医疗工具包装器
# =========================================================================


@dataclass
class MedToolWrapper:
    """包装一个医疗工具（检索/LLM/原生）的统一接口。

    Usage::

        wrapper = MedToolWrapper(
            name="kg_search",
            category="retrieval",
            executor=hybrid_retriever.retrieve,
            on_error_msg="知识图谱暂时不可用",
        )
        result: MedToolResult = wrapper.execute(query="高血压")
    """
    name: str
    category: str                    # "retrieval" | "llm" | "local_tool" | "safety"
    executor: Callable[..., Any]     # 实际的工具函数
    on_error_msg: str = ""           # 失败时面向用户的降级提示
    make_sources: bool = False       # 是否从结果自动提取 SourceTrace
    post_process: Optional[Callable[[MedToolResult], MedToolResult]] = None  # 结果后处理

    def execute(self, config: Optional[HarnessConfig] = None, **params) -> MedToolResult:
        """执行工具，永不抛异常。

        Args:
            config: 可选的 HarnessConfig（使用默认值）。
            **params: 传递给工具函数的参数。

        Returns:
            MedToolResult（永远不抛异常）。
        """
        cfg = config or HarnessConfig()
        timeout = cfg.get_timeout(self.name)
        retry_policy = cfg.get_retry_policy(self.category)

        last_result: Optional[MedToolResult] = None
        start = time.monotonic()

        for attempt in range(retry_policy["max_retries"] + 1):
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(self.executor, **params)
            try:
                data = future.result(timeout=timeout / 1000)

                elapsed = (time.monotonic() - start) * 1000
                result = self._build_success_result(data, elapsed)
                if self.post_process:
                    result = self.post_process(result)
                result.retry_count = attempt
                return result

            except FutureTimeout:
                future.cancel()
                elapsed = (time.monotonic() - start) * 1000
                last_result = MedToolResult(
                    status=ToolStatus.TIMEOUT,
                    tool_name=self.name,
                    error=f"Timeout after {timeout}ms",
                    user_facing_warning=self.on_error_msg or f"「{self.name}」查询超时，结果可能不完整。",
                    confidence=Confidence.LOW,
                    latency_ms=elapsed,
                )
                if attempt < retry_policy["max_retries"]:
                    _sleep(retry_policy["delay_ms"] / 1000 * (retry_policy["backoff"] ** attempt))
                    continue
                break

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                last_result = MedToolResult(
                    status=ToolStatus.FAILED,
                    tool_name=self.name,
                    error=str(exc),
                    user_facing_warning=self.on_error_msg or "该服务暂时不可用。",
                    latency_ms=elapsed,
                )
                if attempt < retry_policy["max_retries"]:
                    _sleep(retry_policy["delay_ms"] / 1000 * (retry_policy["backoff"] ** attempt))
                    continue
                break

            finally:
                pool.shutdown(wait=False)

        return last_result if last_result is not None else MedToolResult.failed(
            tool_name=self.name,
            error="Unknown failure",
            user_facing_warning=self.on_error_msg,
        )

    def _build_success_result(self, data: Any, latency_ms: float) -> MedToolResult:
        """从原始工具返回值构建 MedToolResult。"""
        sources: List[SourceTrace] = []

        if self.make_sources and isinstance(data, dict):
            for key, stype in [("kg_results", "kg"), ("qa_results", "qa"),
                               ("es_results", "es"), ("case_results", "case")]:
                items = data.get(key, [])
                for i, item in enumerate(items):
                    content = item.get("answer") or item.get("text") or str(item)
                    sources.append(SourceTrace(
                        source_type=stype,
                        content=str(content)[:500],
                        score=item.get("rrf_score", item.get("score", 0)),
                        source_name=item.get("source", stype),
                        rank=i + 1,
                    ))

        if sources:
            unique_types = set(s.source_type for s in sources)
            avg_score = sum(s.score for s in sources) / len(sources)
            if len(unique_types) >= 2 and avg_score > 0.3:
                confidence = Confidence.HIGH
            else:
                confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.HIGH

        return MedToolResult.ok(
            tool_name=self.name,
            data=data,
            sources=sources,
            confidence=confidence,
            latency_ms=latency_ms,
        )


def _sleep(seconds: float) -> None:
    """Thread-safe sleep (可 mock 的延迟函数)。"""
    import time as _time
    _time.sleep(max(seconds, 0))


# =========================================================================
# 工厂函数：为具体 MedAgent 组件创建包装器
# =========================================================================


def create_retrieval_wrapper(retriever) -> MedToolWrapper:
    """包装 HybridRetriever → MedToolWrapper。

    负责执行多源检索并将结果解析为带 SourceTrace 的 MedToolResult。
    """
    return MedToolWrapper(
        name="hybrid_retrieval",
        category="retrieval",
        executor=retriever.retrieve,
        make_sources=True,
        on_error_msg="多源检索暂时不可用，部分结果可能缺失。",
    )


def create_llm_wrapper(answer_generator) -> MedToolWrapper:
    """包装 AnswerGenerator → MedToolWrapper。"""
    return MedToolWrapper(
        name="llm_generate",
        category="llm",
        executor=answer_generator.generate,
        on_error_msg="AI 模型暂时无法生成回答，请稍后重试。",
    )


def create_safety_wrapper(safety_guard) -> MedToolWrapper:
    """包装 SafetyGuard → MedToolWrapper。

    返回的 MedToolResult 会在 ``risk_flags`` 中携带检测到的风险关键词。
    """

    def _detect_and_append(query: str, answer: str = "",
                           retrieval_quality: str = "none",
                           query_type: str = "") -> dict:
        risk_info = safety_guard.detect_risk(query)
        full_answer = safety_guard.append_safety_notice(
            answer, risk_info,
            retrieval_quality=retrieval_quality,
            query_type=query_type,
        )
        return {
            "answer": full_answer,
            "has_risk": risk_info.get("has_risk", False),
            "risk_keywords": risk_info.get("risk_keywords", []),
            "risk_level": risk_info.get("level", "none"),
        }

    def _extract_risk_flags(result: MedToolResult) -> MedToolResult:
        if isinstance(result.data, dict) and result.data.get("has_risk"):
            result.risk_flags = result.data.get("risk_keywords", [])
        return result

    return MedToolWrapper(
        name="safety_check",
        category="safety",
        executor=_detect_and_append,
        post_process=_extract_risk_flags,
    )


def create_builtin_wrapper(builtin_tool) -> MedToolWrapper:
    """包装 BaseTool（剂量计算/正常值/科室导诊）→ MedToolWrapper。"""
    tool_name = getattr(builtin_tool, 'name', builtin_tool.__class__.__name__)
    return MedToolWrapper(
        name=tool_name,
        category="local_tool",
        executor=builtin_tool.execute,
    )
