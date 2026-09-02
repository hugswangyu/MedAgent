"""通话开始前渲染固定 SessionSystemPrompt。"""

from __future__ import annotations

from dataclasses import dataclass

from medlive.config.settings import RagToolMode
from medlive.context.defaults import (
    DEFAULT_RAG_TOOL_DESCRIPTION,
    PHASE0_MANDATORY_RULES,
    RAG_DISABLED_DESCRIPTION,
)
from medlive.context.store import ContextStore


@dataclass(slots=True)
class SessionPromptRenderResult:
    """一次通话固定系统提示词渲染结果。"""

    prompt: str
    prompt_chars: int
    rag_tool_mode: RagToolMode
    history_count: int
    kb_id: str
    kb_name: str


class SessionPromptRenderer:
    """把模板、SOUL 和知识库概览渲染成本次通话的固定提示词。"""

    def __init__(self, *, store: ContextStore, history_limit: int) -> None:
        """绑定存储和历史读取数量。"""

        self.store = store
        self.history_limit = history_limit

    def render(
        self,
        *,
        kb_id: str,
        kb_name: str,
        rag_tool_mode: RagToolMode,
    ) -> SessionPromptRenderResult:
        """渲染并落盘本次通话固定系统提示词。"""

        history: list[dict] = []
        rendered_history = "独立 history 已停用；本次会话仅逐轮使用 PostgreSQL confirmed 记忆。"
        prompt = self.store.read_system_prompt_template()
        prompt = prompt.replace("{{SOUL_MD}}", self.store.read_soul().strip() or "无。")
        prompt = prompt.replace("{{HISTORY_JSONL}}", rendered_history)
        prompt = prompt.replace("{{KNOWLEDGE_OVERVIEW_MD}}", self.store.read_knowledge_overview(kb_id).strip())
        prompt = prompt.replace("{{RAG_TOOL_DESCRIPTION}}", self._rag_tool_description(rag_tool_mode))
        prompt = prompt.replace("{{KB_ID}}", kb_id)
        prompt = prompt.replace("{{KB_NAME}}", kb_name)
        prompt = f"{prompt.rstrip()}\n\n{PHASE0_MANDATORY_RULES.strip()}\n"
        self.store.write_session_system_prompt(prompt)
        return SessionPromptRenderResult(
            prompt=prompt,
            prompt_chars=len(prompt),
            rag_tool_mode=rag_tool_mode,
            history_count=len(history),
            kb_id=kb_id,
            kb_name=kb_name,
        )

    @staticmethod
    def _rag_tool_description(rag_tool_mode: RagToolMode) -> str:
        """根据 RAG 模式返回工具说明。"""

        if rag_tool_mode == "auto":
            return DEFAULT_RAG_TOOL_DESCRIPTION.strip()
        return RAG_DISABLED_DESCRIPTION.strip()
