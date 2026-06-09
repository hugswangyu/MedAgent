"""答案生成器：封装 LLM 调用以生成最终回答。

支持缓存（MD5 hash + LRU）和主 API 失败时回退到 Ollama。
"""

import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional

from medrag.config.settings import settings
from medrag.llm import get_llm_provider
from medrag.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """调用配置好的 LLM，根据提示词生成最终回答。

    用法::

        generator = AnswerGenerator()
        answer = generator.generate(prompt)

        # 注入自定义 provider:
        generator = AnswerGenerator(llm_provider=get_llm_provider("ollama"))
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        cache_max_size: int | None = None,
        cache_ttl_seconds: int | None = None,
    ):
        self._provider = llm_provider or get_llm_provider()
        self._client = self._provider.client
        self._model = self._provider.default_model
        self._fallback_provider: Optional[LLMProvider] = None

        self._cache_max_size = cache_max_size if cache_max_size is not None else settings.llm_cache_size
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds if cache_ttl_seconds is not None else settings.llm_cache_ttl)
        self._cache: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _as_messages(prompt_or_messages: str | list[dict]) -> list[dict]:
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]
        return prompt_or_messages

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(model: str, messages: list) -> str:
        raw = f"{model}:{json.dumps(messages, sort_keys=True, ensure_ascii=False)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if key in self._cache:
            ts, response = self._cache[key]
            if datetime.now() - ts < self._cache_ttl:
                return response
            del self._cache[key]
        return None

    def _cache_set(self, key: str, response: str) -> None:
        if len(self._cache) >= self._cache_max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (datetime.now(), response)

    # ------------------------------------------------------------------
    # 回退
    # ------------------------------------------------------------------

    def _try_fallback(self, messages: list, model: str | None) -> Optional[str]:
        if self._fallback_provider is None:
            try:
                self._fallback_provider = get_llm_provider("ollama")
            except Exception:
                return None
        try:
            resp = self._fallback_provider.client.chat.completions.create(
                model=model or self._fallback_provider.default_model,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def generate(self, prompt_or_messages: str | list[dict], model: str | None = None) -> str:
        messages = self._as_messages(prompt_or_messages)
        model_name = model or self._model
        cache_key = self._make_cache_key(model_name, messages)

        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("LLM cache hit for %s", model_name)
            return cached

        last_exc = None

        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            answer = response.choices[0].message.content or ""
            self._cache_set(cache_key, answer)
            return answer
        except Exception as exc:
            last_exc = exc
            logger.warning("Primary LLM %s failed: %s", self._provider.name, exc)

        fallback_answer = self._try_fallback(messages, model_name)
        if fallback_answer is not None:
            logger.info("Fell back to Ollama for LLM call")
            self._cache_set(cache_key, fallback_answer)
            return fallback_answer

        return (
            f"抱歉，调用 {self._provider.name} 生成回答时出错：{last_exc}\n"
            f"已尝试本地 Ollama 回退但仍然失败。请检查网络连接或重试。"
        )

    def generate_stream(self, prompt_or_messages: str | list[dict], model: str | None = None):
        messages = self._as_messages(prompt_or_messages)
        model_name = model or self._model

        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
            return
        except Exception as exc:
            logger.warning("Primary LLM stream failed: %s", exc)
            _stream_err = exc

        fallback = self._try_fallback(messages, model_name)
        if fallback is not None:
            yield fallback
            return
        yield f"\n[流式生成错误: {_stream_err}]"
