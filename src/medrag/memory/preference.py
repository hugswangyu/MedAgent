"""User preference store — medical-aware key-value pairs extracted from conversation.

Two extraction paths:
  1. LLM-based (primary): calls DeepSeek to extract structured medical preferences.
  2. Rule-based (fallback): regex patterns for Chinese medical facts, used when LLM
     is unavailable or returns empty.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM extraction prompt (medical-focused)
# ---------------------------------------------------------------------------

_PREFERENCE_EXTRACT_PROMPT = (
    "你是一个医疗信息提取器。从用户的对话消息中提取用户的个人健康信息。\n\n"
    "只提取用户明确陈述的医学事实，不要猜测、推断或补充。\n\n"
    "重点关注以下类别：\n"
    "- 过敏史：药物、食物、环境过敏原\n"
    "- 慢性病史：已确诊的慢性疾病和确诊时间\n"
    "- 长期用药：正在长期服用的药物名称和剂量\n"
    "- 手术史：做过的手术和时间\n"
    "- 体征指标：血压、血糖、体重等可量化的检查结果\n"
    '- 家族史：直系亲属的重大疾病\n\n'
    '返回 JSON 格式，preferences 数组的每一项是 ["中文类别名", "提取到的内容"]。'
    '如果没有找到任何健康信息，返回 {"preferences": []}。\n\n'
    "示例：\n"
    "用户：我对青霉素过敏，有高血压，一直在吃硝苯地平\n"
    '输出：{"preferences": [["过敏史", "青霉素过敏"], ["慢性病史", "高血压"], ["长期用药", "硝苯地平"]]}\n\n'
    "用户：我做过胆囊切除手术\n"
    '输出：{"preferences": [["手术史", "胆囊切除手术"]]}\n\n'
    "用户：我血压最近150/95，控制得不太好\n"
    '输出：{"preferences": [["体征指标", "血压150/95，控制不佳"]]}\n\n'
    "用户消息：{text}\n\n"
    "请只输出 JSON，不要加任何其他文字："
)

# ---------------------------------------------------------------------------
# Fallback regex rules (medical domain)
# ---------------------------------------------------------------------------

_MEDICAL_RULES: List[Tuple[re.Pattern, str]] = [
    # 过敏史 — capture group(1) = "X过敏"
    (re.compile(r"(?:对|吃|打|服用|注射|输)\s*(.{1,18}?过敏)"), "过敏史"),
    # 慢性病史 — capture group(1) = disease name
    (re.compile(
        r"(?:患有?|得了?|确诊|有|既往|病史)\s*"
        r"(高血压|糖尿病|冠心病|哮喘|慢阻肺|慢性支气管炎|乙肝|甲亢|甲减"
        r"|慢性肾炎|心脏病|痛风|高血脂|高尿酸|脑梗|脑出血|心梗|心衰|房颤"
        r"|肝硬化|脂肪肝|类风湿|强直|银屑病|贫血)"
    ), "慢性病史"),
    # 长期用药 — capture group(1) = drug name
    (re.compile(
        r"(?:在吃|在服用|长期吃|一直吃|每天吃|每天服用|长期服用|规律服用)\s*"
        r"(.{1,20}?)(?:[，。\.！,，\s]|$)"
    ), "长期用药"),
    # 手术史 — capture group(1) = full procedure description
    (re.compile(
        r"(?:做过?|动过|接受过|行)\s*(.{1,25}?(?:手术|切除|支架|搭桥|置换|移植))"
    ), "手术史"),
    # 体征指标 — capture group(1) = value
    (re.compile(r"血压\s*(?:是|为|有|在)?\s*(\d{2,3}\s*/?\s*\d{2,3})(?:\s*(?:mmHg|毫米汞柱))?"), "体征指标"),
    (re.compile(r"血糖[^\d]*(\d+(?:\.\d+)?)(?:\s*(?:mmol/L|毫摩尔每升))?"), "体征指标"),
]


class PreferenceStore:
    """Medical preference key-value store.

    LLM extraction is primary; regex rules serve as fallback when LLM API
    is unreachable or returns no results.

    Persisted to PostgreSQL for multi-tenant isolation.
    """

    def __init__(self, username: str = ""):
        self._username = username
        self._data: Dict[str, str] = {}
        if username:
            self._load_from_pg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _save_to_pg(self, key: str, value: str) -> None:
        if not self._username:
            return
        try:
            from medrag.infrastructure.storage.postgres_client import pref_save
            pref_save(self._username, key, value)
        except Exception:
            logger.debug("Failed to persist preference", exc_info=True)

    def _load_from_pg(self) -> None:
        try:
            from medrag.infrastructure.storage.postgres_client import pref_load_all
            self._data = pref_load_all(self._username)
        except Exception:
            logger.debug("Failed to load preferences from PG", exc_info=True)

    def save(self, key: str, value: str) -> None:
        if key and value:
            self._data[key] = value
            self._save_to_pg(key, value)

    def save_batch(self, kvs: Dict[str, str]) -> None:
        for k, v in kvs.items():
            if k and v:
                self._data[k] = v
                self._save_to_pg(k, v)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(key, default)

    def all(self) -> Dict[str, str]:
        return dict(self._data)

    # ------------------------------------------------------------------
    # Extraction (LLM first, regex fallback)
    # ------------------------------------------------------------------

    def llm_extract(self, text: str) -> bool:
        """Extract preferences via DeepSeek. Falls back to regex on failure.

        Returns True if LLM extraction succeeded and produced results.
        """
        if not self._has_llm():
            logger.debug("No DeepSeek API key configured, using regex fallback")
            return self._regex_fallback(text)

        try:
            from openai import OpenAI
            from medrag.config.settings import settings

            client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
            prompt = _PREFERENCE_EXTRACT_PROMPT.format(text=text)
            response = client.chat.completions.create(
                model=settings.deepseek_intent_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content
            if not raw:
                return self._regex_fallback(text)

            data = json.loads(raw)
            pairs = data.get("preferences", [])
            if not pairs:
                return self._regex_fallback(text)

            for pair in pairs:
                if len(pair) == 2 and pair[0] and pair[1]:
                    self.save(str(pair[0]).strip(), str(pair[1]).strip())
            return True

        except Exception:
            logger.debug("LLM preference extraction failed, falling back to regex", exc_info=True)
            return self._regex_fallback(text)

    def _regex_fallback(self, text: str) -> bool:
        """Run medical regex rules against *text*. Returns True if any matched."""
        matched = False
        for pattern, key in _MEDICAL_RULES:
            m = pattern.search(text)
            if m:
                value = m.group(1).strip()
                self.save(key, value)
                matched = True
        return matched

    @staticmethod
    def _has_llm() -> bool:
        try:
            from medrag.config.settings import settings
            return bool(settings.deepseek_api_key)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context(self) -> str:
        """Format stored preferences as a context string for LLM prompts.

        Returns empty string if no preferences stored.
        """
        if not self._data:
            return ""
        lines = [f"{k}: {v}" for k, v in self._data.items()]
        return "【用户偏好】\n" + "\n".join(lines)

    @property
    def data(self) -> Dict[str, str]:
        return self._data
