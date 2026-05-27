"""安全卫士：医疗风险检测与安全提示注入。"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# 高风险关键词 → 类别标签
# ---------------------------------------------------------------------------

_HIGH_RISK_KEYWORDS: Dict[str, str] = {
    "胸痛": "胸痛",
    "呼吸困难": "呼吸困难",
    "意识不清": "意识不清",
    "抽搐": "抽搐",
    "大出血": "大出血",
    "便血": "便血",
    "黑便": "黑便",
    "高热不退": "高热不退",
    "剧烈腹痛": "剧烈腹痛",
    "孕妇": "孕妇",
    "婴儿": "婴儿",
    "自杀": "自杀",
    "过量服药": "过量服药",
    "休克": "休克",
}

_HIGH_RISK_WARNING = (
    "你描述的情况可能存在较高风险，建议尽快线下就医或急诊评估。"
)

_DISCLAIMER = (
    "以上内容仅用于健康科普和就医参考，不能替代医生面诊。"
)


class SafetyGuard:
    """检测高风险医疗查询并注入安全提示。

    用法::

        guard = SafetyGuard()
        risk = guard.detect_risk(query, answer)
        safe_answer = guard.append_safety_notice(answer, risk)
    """

    def detect_risk(self, query: str, answer: str = "") -> Dict:
        """扫描 *query* 和 *answer* 中的高风险关键词。

        Args:
            query: 用户问题。
            answer: LLM 生成的回答（可选）。一并扫描，避免遗漏
                    模型自身回复中出现的高风险关键词。

        Returns:
            字典，包含:
            - ``is_high_risk``: 匹配到任何风险关键词时为 ``True``。
            - ``risk_types``: 匹配到的关键词标签列表。
            - ``safety_message``: 高风险警告字符串，或 ``""``。
        """
        combined = f"{query}\n{answer}"
        risk_types: List[str] = []

        for keyword, label in _HIGH_RISK_KEYWORDS.items():
            if keyword in combined:
                risk_types.append(label)

        is_high_risk = len(risk_types) > 0
        return {
            "is_high_risk": is_high_risk,
            "risk_types": risk_types,
            "safety_message": _HIGH_RISK_WARNING if is_high_risk else "",
        }

    @staticmethod
    def append_safety_notice(answer: str, risk_info: Dict) -> str:
        """根据 *risk_info* 向 *answer* 中注入安全提示。

        - 若为高风险，在开头添加紧急就医警告。
        - 始终在末尾添加免责声明。

        Args:
            answer: 原始回答文本。
            risk_info: :meth:`detect_risk` 返回的字典。

        Returns:
            注入了安全提示的回答文本。
        """
        parts: List[str] = []

        if risk_info.get("is_high_risk"):
            parts.append(risk_info.get("safety_message", _HIGH_RISK_WARNING))

        parts.append(answer.strip())

        disclaimer = _DISCLAIMER
        if not answer.strip().endswith(disclaimer):
            parts.append(disclaimer)

        return "\n\n".join(parts)
