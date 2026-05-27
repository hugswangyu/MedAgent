"""Safety Guard: medical risk detection and safety notice injection."""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# High-risk keyword → category label
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
    """Detect high-risk medical queries and inject safety notices.

    Usage::

        guard = SafetyGuard()
        risk = guard.detect_risk(query, answer)
        safe_answer = guard.append_safety_notice(answer, risk)
    """

    def detect_risk(self, query: str, answer: str = "") -> Dict:
        """Scan *query* and *answer* for high-risk keywords.

        Args:
            query: The user's question.
            answer: The LLM-generated answer (optional). Also scanned so
                    that risk keywords appearing in the model's own
                    response are not overlooked.

        Returns:
            A dict with:
            - ``is_high_risk``: ``True`` when any risk keyword is found.
            - ``risk_types``: list of matched keyword labels.
            - ``safety_message``: the high-risk warning string, or ``""``.
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
        """Apply safety notices to *answer* based on *risk_info*.

        - If high-risk, prepend the urgent-care warning.
        - Always append the disclaimer.

        Args:
            answer: The original answer text.
            risk_info: Dict returned by :meth:`detect_risk`.

        Returns:
            The answer text with safety notices injected.
        """
        parts: List[str] = []

        if risk_info.get("is_high_risk"):
            parts.append(risk_info.get("safety_message", _HIGH_RISK_WARNING))

        parts.append(answer.strip())

        disclaimer = _DISCLAIMER
        if not answer.strip().endswith(disclaimer):
            parts.append(disclaimer)

        return "\n\n".join(parts)
