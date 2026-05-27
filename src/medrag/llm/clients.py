"""统一 LLM 客户端工厂。

所有 LLM 后端（DeepSeek、ZhipuAI、Ollama）的单一创建入口。
客户端按 provider 缓存，重复调用返回同一实例。
"""

from __future__ import annotations

from medrag.config.settings import settings

_clients: dict[str, object] = {}


def get_llm_client(provider: str | None = None):
    """返回 *provider* 对应的缓存 LLM 客户端。

    Args:
        provider: ``"deepseek"`` | ``"zhipuai"`` | ``"ollama"``。
                  默认为 ``settings.llm_provider``。

    Returns:
        兼容 OpenAI 的客户端（``openai.OpenAI`` 或 ``zhipuai.ZhipuAI``）。
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
