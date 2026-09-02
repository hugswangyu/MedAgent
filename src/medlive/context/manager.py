"""当前通话消息记录和 RAG 工具查询管理。"""

from __future__ import annotations

from typing import ClassVar

from medlive.agent.tool import RagClient, RagQueryResult
from medlive.context.store import ContextStore


class ContextManager:
    """通话中只负责消息落盘、追问改写和 RAG 工具调用。"""

    _FOLLOWUP_PHRASES: ClassVar[set[str]] = {
        "接着说",
        "继续",
        "继续说",
        "详细说",
        "详细说说",
        "展开说说",
        "展开讲讲",
        "然后呢",
        "还有呢",
        "再说说",
        "再讲讲",
        "具体点",
        "讲详细点",
        "说详细点",
    }

    def __init__(self, *, store: ContextStore, rag_client: RagClient) -> None:
        """绑定上下文依赖。"""

        self.store = store
        self.rag_client = rag_client
        self._last_user_text = ""
        self._previous_user_text = ""
        self._last_rag_query = ""

    def record_user_message(self, *, content: str, turn_index: int) -> None:
        """记录用户输入并维护短追问锚点。"""

        text = content.strip()
        if not text:
            return
        if text != self._last_user_text:
            self._previous_user_text = self._last_user_text
        self._last_user_text = text
        self.store.append_message(role="user", content=text, turn_index=turn_index)
        self._write_runtime_state(turn_index=turn_index, rag_result=None)

    async def query_knowledge_base(
        self,
        *,
        query: str,
        original_query: str,
        turn_index: int,
        source: str,
        tool_name: str | None = None,
    ) -> RagQueryResult:
        """通过当前锁定知识库执行 RAG 工具查询。"""

        rag_query = self._build_rag_query(query)
        result = await self.rag_client.query_context(
            query=rag_query,
            original_query=original_query,
            last_query=self._last_rag_query or self._previous_user_text or None,
            source=source,
            tool_name=tool_name,
            turn_index=turn_index,
        )
        self._write_runtime_state(turn_index=turn_index, rag_result=result)
        return result

    def record_assistant_message(self, *, content: str, turn_index: int) -> None:
        """记录助手回复和回答长度观测字段。"""

        char_count = len(content.strip())
        rag_records = [item for item in self.store.read_rag_context() if item.get("turn_index") == turn_index]
        used_rag = bool(rag_records)
        too_long = char_count > 180
        metadata = {
            "char_count": char_count,
            "tts_text_chars": char_count,
            "tts_text_chars_source": "assistant_text",
            "too_long": too_long,
            "used_rag": used_rag,
            "rag_tool_mode": self.rag_client.settings.rag_tool_mode,
        }
        self.store.append_message(
            role="assistant",
            content=content,
            turn_index=turn_index,
            metadata=metadata,
        )
        state = self.store.read_runtime_state()
        state.update(
            {
                "last_assistant_chars": char_count,
                "last_tts_text_chars": char_count,
                "last_tts_text_chars_source": "assistant_text",
                "last_answer_too_long": too_long,
                "last_answer_used_rag": used_rag,
                "rag_tool_mode": self.rag_client.settings.rag_tool_mode,
            }
        )
        self.store.write_runtime_state(state)

    def _build_rag_query(self, user_text: str) -> str:
        """为短追问补上上一轮主题。"""

        query = user_text.strip()
        if not self._is_followup_query(query):
            self._last_rag_query = query
            return query
        anchor = self._last_rag_query or self._previous_user_text
        if not anchor:
            return query
        return f"上一轮问题：{anchor}\n当前追问：{query}\n请围绕上一轮主题继续补充。"

    @classmethod
    def _is_followup_query(cls, user_text: str) -> bool:
        """判断用户输入是否是短追问。"""

        text = user_text.strip()
        return bool(text and len(text) <= 12 and any(phrase in text for phrase in cls._FOLLOWUP_PHRASES))

    def _write_runtime_state(self, *, turn_index: int, rag_result: RagQueryResult | None) -> None:
        """写入当前运行态，便于前端和排障读取。"""

        state = self.store.read_runtime_state()
        state.update(
            {
                "turn_index": turn_index,
                "last_user_text": self._last_user_text,
                "previous_user_text": self._previous_user_text,
                "last_rag_query": self._last_rag_query,
                "rag_tool_mode": self.rag_client.settings.rag_tool_mode,
            }
        )
        if rag_result is not None:
            state["last_rag"] = {
                "hit": rag_result.hit,
                "has_context": bool(rag_result.context_block),
                "request_id": rag_result.request_id,
                "metrics": rag_result.metrics,
                "error": rag_result.error,
            }
        self.store.write_runtime_state(state)
