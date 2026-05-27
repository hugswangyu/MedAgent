"""Answer Generator: encapsulates LLM calls for final answer generation."""

from __future__ import annotations

from medrag.config.settings import settings
from medrag.llm import get_llm_client


class AnswerGenerator:
    """Call the configured LLM to generate a final answer from a prompt.

    Usage::

        generator = AnswerGenerator()
        answer = generator.generate(prompt)
    """

    def __init__(self, provider: str | None = None):
        """
        Args:
            provider: One of ``"deepseek"``, ``"zhipuai"``, ``"ollama"``.
                      Defaults to ``settings.llm_provider``.
        """
        self.provider = (provider or settings.llm_provider).strip().lower()

        if self.provider == "deepseek":
            self._model = settings.deepseek_answer_model
        elif self.provider == "zhipuai":
            self._model = settings.zhipuai_model
        elif self.provider == "ollama":
            self._model = settings.ollama_model
        else:
            raise ValueError(
                f"不支持的 LLM_PROVIDER: {self.provider!r}，"
                f"可选值为 deepseek / zhipuai / ollama"
            )

        self._client = get_llm_client(self.provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the generated answer text.

        Args:
            prompt: The full prompt string (typically from
                    :meth:`PromptBuilder.build_answer_prompt`).

        Returns:
            The LLM response text, or a friendly error message on failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        except Exception as exc:
            provider = self.provider
            return (
                f"抱歉，调用 {provider} 生成回答时出错：{exc}\n"
                f"请检查 API Key 是否正确、网络是否通畅。"
            )
