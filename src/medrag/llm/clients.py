"""Unified LLM client factory.

Single point of creation for all LLM backends (DeepSeek, ZhipuAI, Ollama).
Clients are cached per provider so repeated calls return the same instance.
"""

from __future__ import annotations

from medrag.config.settings import settings

_clients: dict[str, object] = {}


def get_llm_client(provider: str | None = None):
    """Return a cached LLM client for *provider*.

    Args:
        provider: ``"deepseek"`` | ``"zhipuai"`` | ``"ollama"``.
                  Defaults to ``settings.llm_provider``.

    Returns:
        An OpenAI-compatible client (``openai.OpenAI`` or ``zhipuai.ZhipuAI``).
    """
    provider = (provider or settings.llm_provider).strip().lower()

    if provider not in _clients:
        if provider == "deepseek":
            _clients[provider] = _create_deepseek()
        elif provider == "zhipuai":
            _clients[provider] = _create_zhipuai()
        elif provider == "ollama":
            _clients[provider] = _create_ollama()
        else:
            raise ValueError(
                f"不支持的 LLM_PROVIDER: {provider!r}，"
                f"可选值为 deepseek / zhipuai / ollama"
            )

    return _clients[provider]


def _create_deepseek():
    from openai import OpenAI
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


def _create_zhipuai():
    from zhipuai import ZhipuAI
    return ZhipuAI(api_key=settings.zhipuai_api_key)


def _create_ollama():
    from openai import OpenAI
    return OpenAI(
        api_key="ollama",
        base_url=settings.ollama_base_url,
    )
