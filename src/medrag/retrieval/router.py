"""Hybrid query router for the medical QA system.

Routes a user question to the appropriate retrieval source(s):
  - neo4j_kg      structured medical KG (diseases, symptoms, drugs, diet, etc.)
  - toyhom_qa     Toyhom medical QA vector store

Supports two modes:
  - LLM routing (default): semantic classification with rule fallback.
  - Rule routing: fast keyword matching for environments without an LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from medrag.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based routing tables  (fallback & llm_client-less mode)
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[list[str], str, bool, bool]] = [
    # --- medication ---
    (
        ["吃什么药", "用什么药", "药品", "药物", "药", "剂量", "用法用量",
         "不能吃什么药", "忌什么药", "生产商", "哪个厂"],
        "medication",
        True, True,
    ),
    # --- diet ---
    (
        ["吃什么", "不能吃什么", "宜吃", "忌吃", "饮食", "食谱",
         "忌口", "能吃什么", "不能吃什么"],
        "diet",
        True, True,
    ),
    # --- department ---
    (
        ["挂什么科", "看什么科", "什么科室", "去哪个科", "哪个科",
         "科室", "挂号"],
        "department",
        True, True,
    ),
    # --- test / examination ---
    (
        ["检查", "做什么检查", "怎么查", "体检", "筛查"],
        "test_report",
        True, True,
    ),
    # --- disease facts ---
    (
        ["并发症", "并发", "引起什么", "导致什么", "预防", "治愈",
         "治疗周期", "能治好吗", "会死吗", "严重吗", "遗传吗",
         "传染吗", "易感人群", "什么人容易", "病因", "原因",
         "怎么引起的", "为什么会", "简介", "什么是", "是什么病"],
        "disease_fact",
        True, True,
    ),
    # --- symptom consultation ---
    (
        ["症状", "表现", "什么感觉", "征兆", "怎么知道",
         "是不是得了", "怎么判断", "确诊"],
        "symptom_consult",
        True, True,
    ),
    # --- treatment ---
    (
        ["治疗", "怎么办", "怎么治", "如何治", "治愈", "治好",
         "手术", "住院", "康复", "怎么处理", "如何缓解"],
        "disease_fact",
        True, True,
    ),
]

_FALLBACK_QUERY_TYPE = "general_medical_qa"

# Valid query_type values (used by both rule and LLM routing)
QUERY_TYPES = [
    "disease_fact",
    "symptom_consult",
    "medication",
    "test_report",
    "diet",
    "department",
    "general_medical_qa",
]

# ---------------------------------------------------------------------------
# LLM routing prompt  (kept short to minimise latency)
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """你是一个医疗问答路由分类器。你的任务是根据用户问题，判断应该查询哪些信息源。

可选信息源：
- kg: Neo4j 医学知识图谱（疾病、症状、药品、饮食、科室等结构化知识）
- qa: Toyhom 医疗问答向量库（通用医学问答）

可选 query_type：
- disease_fact: 疾病事实查询（病因、预防、并发症、治愈率、简介等）
- symptom_consult: 症状咨询（有什么症状、是不是得了某病、怎么判断）
- medication: 药品相关（吃什么药、用法用量、药物信息、生产商）
- test_report: 检查/检验相关（做什么检查、体检、筛查）
- diet: 饮食相关（能吃什么、不能吃什么、饮食建议）
- department: 科室咨询（挂什么科、看什么科室）
- general_medical_qa: 泛医疗问题或非医疗问题

规则：
1. 若问题涉及疾病事实、症状、药物、饮食、科室，启用 kg。
2. 绝大多数医疗问题都应启用 qa。
3. 非医疗问题只启用 qa，query_type=general_medical_qa。

请输出以下 JSON 格式（不要加任何其他文字）：
{{"kg": true/false, "qa": true/false, "query_type": "...", "reason": "一句中文解释"}}"""

_ROUTER_USER = """用户问题: {query}

JSON:"""

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class QueryRouter:
    """Hybrid router: LLM-first with rule fallback.

    Usage::

        from openai import OpenAI
        llm = OpenAI(api_key=..., base_url=...)

        router = QueryRouter(llm_client=llm)
        decision = router.route("感冒了怎么办")

        # Force rule mode:
        decision = router.route("...", use_llm=False)

        # No LLM client → auto rule-only:
        router = QueryRouter()
        decision = router.route("...")
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: OpenAI-compatible client.  If None, always uses rules.
        """
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str, use_llm: bool = True) -> Dict:
        """Route *query* to retrieval sources.

        Returns dict with keys:
            use_kg, use_toyhom_qa, reason, query_type
        """
        if use_llm and self.llm is not None:
            result = self._llm_route(query)
            if result is not None:
                return result

        return self._rule_route(query)

    # ------------------------------------------------------------------
    # LLM routing
    # ------------------------------------------------------------------

    def _llm_route(self, query: str) -> Optional[Dict]:
        """Try LLM-based routing.  Returns None on any failure."""
        try:
            response = self.llm.chat.completions.create(
                model=settings.deepseek_default_model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM},
                    {"role": "user",
                     "content": _ROUTER_USER.format(query=query)},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content
            return self._parse_llm_response(raw)
        except Exception:
            logger.debug("LLM routing failed, falling back to rules", exc_info=True)
            return None

    @staticmethod
    def _parse_llm_response(raw: str) -> Optional[Dict]:
        """Parse LLM JSON output, translate short keys, validate."""
        if not raw:
            return None
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("\n", 1)[0]
            data = json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{[^{}]*\}', raw)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
            else:
                return None

        canonical: Dict = {
            "use_kg": data.get("kg", data.get("use_kg", False)),
            "use_toyhom_qa": data.get("qa", data.get("use_toyhom_qa", False)),
            "reason": data.get("reason", ""),
            "query_type": data.get("query_type", ""),
        }

        if canonical["query_type"] not in QUERY_TYPES:
            logger.debug("LLM route invalid query_type: %s", canonical["query_type"])
            return None

        # Ensure at least one source is on
        if not (canonical["use_kg"] or canonical["use_toyhom_qa"]):
            canonical["use_toyhom_qa"] = True

        return canonical

    # ------------------------------------------------------------------
    # Rule-based routing (fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_route(query: str) -> Dict:
        for keywords, qtype, kg, toyhom in _ROUTES:
            if any(kw in query for kw in keywords):
                reason = _build_reason(kg, toyhom, keywords, query)
                return {
                    "use_kg": kg,
                    "use_toyhom_qa": toyhom,
                    "reason": reason,
                    "query_type": qtype,
                }

        return {
            "use_kg": False,
            "use_toyhom_qa": True,
            "reason": "未匹配到特定规则，默认使用通用问答库",
            "query_type": _FALLBACK_QUERY_TYPE,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_reason(kg: bool, toyhom: bool, keywords: list[str], query: str) -> str:
    parts: list[str] = []
    matched = [kw for kw in keywords if kw in query]
    tag = "、".join(matched[:3])
    if kg:
        parts.append(f"命中知识图谱关键词「{tag}」→ 开启 neo4j_kg")
    if toyhom:
        parts.append(f"命中关键词「{tag}」→ 开启 toyhom_qa")
    return "；".join(parts) if parts else f"关键词匹配: {tag}"
