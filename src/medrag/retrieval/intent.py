"""医疗查询意图识别。

从 KGRetriever 中提取，可独立于 Neo4j 流水线复用。
通过 few-shot LLM 提示词将用户问题分类到 15 个预定义医学查询意图之一。

意图识别使用独立的 DeepSeek 客户端（不受 LLM_PROVIDER 影响），
确保 KG 检索的意图分类稳定可靠。
"""

from __future__ import annotations

from openai import OpenAI

from medrag.config.settings import settings
from medrag.prompts import FOCUS_PROMPT_TEMPLATE, INTENT_PROMPT_TEMPLATE

_intent_client: OpenAI | None = None

# extract_focus 输出归一化：这些词都视为"问整体"（无具体细分点）
_BROAD_TOKENS = {"无", "没有", "整体", "概况", "全部", "无具体", "none", "null"}


def _get_intent_client() -> OpenAI:
    """获取意图识别专用的 DeepSeek 客户端（延迟初始化，单例）。"""
    global _intent_client
    if _intent_client is None:
        _intent_client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _intent_client


def recognize_intents(query: str, llm_client=None) -> str:
    """调用 DeepSeek 进行意图识别。

    Args:
        query: 自然语言医学问题。
        llm_client: 已废弃，保留参数仅用于向后兼容。

    Returns:
        原始 API 响应字符串（如 ``["查询疾病简介","查询疾病病因"] # 注释``），
        失败时返回 ``""``。
    """
    try:
        prompt = INTENT_PROMPT_TEMPLATE.format(query=query)
        client = _get_intent_client()
        response = client.chat.completions.create(
            model=settings.deepseek_intent_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return ""


def extract_focus(query: str, attribute: str) -> str | None:
    """抽取用户在某长文本属性下询问的具体细分点。

    Args:
        query: 自然语言医学问题。
        attribute: 长文本属性名（如 ``"疾病病因"``）。

    Returns:
        - 具体细分点关键词（如 ``"遗传"``）—— 用户问的是某个子方向；
        - ``""`` —— 用户问的是整体概况（应走离线提纲摘要）；
        - ``None`` —— LLM 调用失败，调用方应回退到正则残差。
    """
    try:
        prompt = FOCUS_PROMPT_TEMPLATE.format(query=query, attribute=attribute)
        client = _get_intent_client()
        response = client.chat.completions.create(
            model=settings.deepseek_intent_model,
            messages=[{"role": "user", "content": prompt}],
        )
        out = (response.choices[0].message.content or "").strip()
    except Exception:
        return None

    # 解析：取首行，去掉标点/前后缀
    out = out.splitlines()[0].strip() if out else ""
    out = out.strip("。.：:、，,「」\"' 　")
    if not out or out.lower() in _BROAD_TOKENS:
        return ""
    # 模型啰嗦/越界 → 当作无明确细分点，交由概括分支处理
    if len(out) > 12:
        return ""
    return out
