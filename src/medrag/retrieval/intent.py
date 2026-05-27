"""医疗查询意图识别。

从 KGRetriever 中提取，可独立于 Neo4j 流水线复用。
通过 few-shot LLM 提示词将用户问题分类到 15 个预定义医学查询意图之一。
"""

from __future__ import annotations

from medrag.config.settings import settings
from medrag.prompts import INTENT_PROMPT_TEMPLATE


def recognize_intents(query: str, llm_client) -> str:
    """调用 LLM 进行意图识别。

    Args:
        query: 自然语言医学问题。
        llm_client: 兼容 OpenAI 的客户端（chat.completions.create）。

    Returns:
        原始 API 响应字符串（如 ``["查询疾病简介","查询疾病病因"] # 注释``），
        失败时返回 ``""``。
    """
    try:
        prompt = INTENT_PROMPT_TEMPLATE.format(query=query)
        response = llm_client.chat.completions.create(
            model=settings.deepseek_default_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return ""
