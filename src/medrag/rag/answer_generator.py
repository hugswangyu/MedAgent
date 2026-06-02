"""答案生成器：封装 LLM 调用以生成最终回答。"""

from __future__ import annotations

from medrag.llm import get_llm_provider
from medrag.llm.provider import LLMProvider


class AnswerGenerator:
    """调用配置好的 LLM，根据提示词生成最终回答。

    用法::

        generator = AnswerGenerator()
        answer = generator.generate(prompt)

        # 注入自定义 provider:
        generator = AnswerGenerator(llm_provider=get_llm_provider("ollama"))
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        self._provider = llm_provider or get_llm_provider()
        self._client = self._provider.client
        self._model = self._provider.default_model

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _as_messages(prompt_or_messages: str | list[dict]) -> list[dict]:
        """确保返回 messages 格式；传入字符串时自动包装为 user 消息。"""
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]
        return prompt_or_messages

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def generate(self, prompt_or_messages: str | list[dict], model: str | None = None) -> str:
        """发送 prompt 到 LLM，返回生成的回答文本。

        Args:
            prompt_or_messages: 提示词字符串 或 messages 列表（来自
                    :meth:`PromptBuilder.build_answer_prompt` /
                    :meth:`PromptBuilder.build_messages`）。
            model: 可选模型名，覆盖构造时绑定的默认模型。

        Returns:
            LLM 的响应文本，失败时返回友好的错误消息。
        """
        try:
            response = self._client.chat.completions.create(
                model=model or self._model,
                messages=self._as_messages(prompt_or_messages),
            )
            return response.choices[0].message.content or ""

        except Exception as exc:
            return (
                f"抱歉，调用 {self._provider.name} 生成回答时出错：{exc}\n"
                f"请检查 API Key 是否正确、网络是否通畅。"
            )

    def generate_stream(self, prompt_or_messages: str | list[dict], model: str | None = None):
        """流式调用 LLM，逐 token yield 文本片段。

        Args:
            prompt_or_messages: 提示词字符串 或 messages 列表。
            model: 可选模型名，覆盖构造时绑定的默认模型。
        """
        try:
            response = self._client.chat.completions.create(
                model=model or self._model,
                messages=self._as_messages(prompt_or_messages),
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            yield f"\n[流式生成错误: {exc}]"
