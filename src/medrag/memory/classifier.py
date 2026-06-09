"""Memory content classifier — rule-based with LLM fallback.

Mirrors AGI-saber classifyMemoryContent() in agent/agent.go.

Medical-adapted: ``get_importance()`` maps category to priority score
so medical facts get higher importance than general chat.
"""

from __future__ import annotations

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Medical importance mapping
# ---------------------------------------------------------------------------
# Each category gets an automatic importance score for LTM storage.
# Medical facts get the highest priority; general chat is skipped.
# ---------------------------------------------------------------------------

CATEGORY_IMPORTANCE = {
    "fact":         0.80,  # medical facts, allergies, diagnoses
    "identity":     0.70,  # user name, demographics
    "episodic":     0.60,  # past events, experiences
    "policy":       0.60,  # constraints, rules
    "preference":   0.50,  # likes, dislikes, habits
    "tool_failure": 0.45,  # tool call errors
    "general":      0.0,   # uncategorized → skip LTM
}


def get_importance(category: str) -> float:
    """Return the default importance for a memory category.

    Medical facts (``fact``) automatically get 0.8 importance,
    while general chat returns 0.0 (skip LTM storage).
    """
    return CATEGORY_IMPORTANCE.get(category, 0.0)


def classify_memory_content(content: str) -> Tuple[str, List[str], str]:
    """Classify memory content into (category, tags, slot_hint).

    Rule-based first; returns ("general", [], "") if no rule matches.
    LLM fallback slot available via ``llm_classify()``.

    Category values (ordered by medical priority):
      fact         — medical facts, diagnoses, allergies (importance 0.80)
      identity     — user name, demographics (importance 0.70)
      episodic     — specific events, experiences (importance 0.60)
      policy       — constraints, rules (importance 0.60)
      preference   — likes, dislikes, habits (importance 0.50)
      tool_failure — tool call errors (importance 0.45)
      general      — uncategorized (importance 0.0 → skipped)
    """
    combined = content

    # Identity
    if _contains_any(combined, ["叫", "名字", "姓名", "我是", "我的"]):
        return "identity", ["name"], "profile"

    # Preference
    if _contains_any(combined, ["喜欢", "偏好", "习惯", "爱好", "讨厌", "不喜欢", "爱喝", "爱吃"]):
        return "preference", ["preference"], "profile"

    # Medical facts (medical QA domain specific)
    if _contains_any(combined, [
        "过敏", "诊断", "病史", "血压", "血糖", "血脂",
        "手术", "住院", "骨折", "肿瘤", "癌",
        "感染", "炎症", "慢性病", "糖尿病", "高血压",
    ]):
        return "fact", ["medical"], "recall"

    # Tool failure
    if _contains_any(combined, ["工具", "失败", "错误", "报错", "异常", "超时"]):
        return "tool_failure", ["tool", "error"], "tool_state"

    # Episodic (specific past events)
    if _contains_any(combined, ["上次", "以前", "曾经", "之前", "过去", "昨天", "前天", "上周"]):
        return "episodic", ["event"], "recall"

    # Policy / constraint
    if _contains_any(combined, ["禁止", "不要", "不能", "必须", "强制", "规则", "限制"]):
        return "policy", ["constraint"], "constraints"

    return "general", [], ""


def llm_classify(content: str) -> Tuple[str, List[str], str]:
    """LLM-based classification placeholder for future integration.

    Mirrors AGI-saber a.llmClassifyMemory() in agent.go.
    """
    return "general", [], ""


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)
