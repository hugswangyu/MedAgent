"""医疗问答系统混合查询路由器。

将用户问题分类以决定检索策略：
  - neo4j_kg      结构化医学知识图谱（疾病、症状、药品、饮食等）
  - qa            cMedQA2 医学问答向量库

支持两种模式：
  - LLM 路由（默认）：语义分类，附带规则回退。
  - 规则路由：快速关键词匹配，适用于没有 LLM 的环境。

返回路由 metadata（query_type、源选择），不再硬选执行模式。
执行模式统一由 HarnessOrchestrator 决定（始终进入 ReAct 循环）。
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from medrag.config.settings import settings

logger = logging.getLogger(__name__)

# 专用 DeepSeek 客户端（与 intent.py 模式一致）
_router_client = None


def _get_router_client():
    global _router_client
    if _router_client is None:
        from openai import OpenAI
        _router_client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _router_client


# ---------------------------------------------------------------------------
# 规则路由表（回退模式 & 无 LLM 客户端模式）
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[list[str], str, bool, bool]] = [
    # --- 药品 ---
    (
        ["吃什么药", "用什么药", "药品", "药物", "剂量", "用法用量",
         "不能吃什么药", "忌什么药", "生产商", "哪个厂", "谁生产",
         "需要吃", "要不要吃", "该吃", "抗生素", "消炎药", "止痛药",
         "继续吃", "还能吃", "能继续用", "停药", "减药", "换药"],
        "medication",
        True, True,
    ),
    # --- 饮食 ---
    (
        ["吃什么", "不能吃什么", "宜吃", "忌吃", "饮食", "食谱",
         "忌口", "能吃什么", "能吃", "可以吃", "不能吃", "少吃", "多吃"],
        "diet",
        True, True,
    ),
    # --- 科室 ---
    (
        ["挂什么科", "看什么科", "什么科室", "去哪个科", "哪个科",
         "科室", "挂号"],
        "department",
        True, True,
    ),
    # --- 疾病事实（含症状识别、治疗方案查询）---
    (
        ["并发症", "并发", "引起什么", "导致什么", "预防", "治愈",
         "治疗周期", "能治好吗", "会死吗", "严重吗", "遗传吗",
         "传染吗", "易感人群", "什么人容易", "病因", "原因",
         "怎么引起的", "为什么会", "简介", "什么是", "是什么病",
         "症状", "表现", "什么感觉", "征兆", "是不是得了", "怎么判断",
         "是不是", "可能是什么", "会不会是", "怎么治疗", "怎么治"],
        "disease_fact",
        True, True,
    ),
    # --- 通用处置（怎么办/检查/报告解读）---
    (
        ["怎么办", "怎么处理", "如何处理", "如何治",
         "手术", "住院", "康复", "如何缓解", "怎么缓解",
         "检查", "做什么检查", "怎么查", "体检", "筛查",
         "代表什么", "说明什么", "指标"],
        "general_medical_qa",
        True, True,
    ),
]

_FALLBACK_QUERY_TYPE = "general_medical_qa"

_ANSWER_STYLE_BY_TYPE = {
    "disease_fact": "fact_short",
    "medication": "medication_safety",
    "diet": "diet_guidance",
    "department": "department_guidance",
    "general_medical_qa": "general_guidance",
}

_CASE_CONTEXT_KEYWORDS = [
    "我", "本人", "我的", "病历", "病例", "报告", "检查单", "化验单",
    "体检", "复查", "住院", "出院", "既往", "用药", "指标",
]

QUERY_TYPES = [
    "disease_fact",
    "medication",
    "diet",
    "department",
    "general_medical_qa",
]

# ---------------------------------------------------------------------------
# LLM 路由提示词
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """你是一个医疗问答分类器。根据用户问题判断信息源和问题类型。

可选信息源：
- kg: Neo4j 医学知识图谱（疾病、症状、药品、饮食、科室等结构化知识）
- qa: cMedQA2 医疗问答向量库（通用医学问答）

可选 query_type（5类，严格按优先级从高到低判断，前面能匹配就不要用后面的）：
- department: 核心意图是"挂什么科/看什么科室"。例：哮喘挂什么科？胃痛该看哪个科？
- medication: 核心意图涉及具体药物——包括：药名查询（谁生产/什么药）、用法用量、是否需要用药（需要吃抗生素吗）、药物安全（能不能继续吃、和其他药冲突吗）。例：阿莫西林是谁生产的？嗓子疼需要吃抗生素吗？二甲双胍肾功能异常还能继续吃吗？
- diet: 核心意图是食物/饮食——包括：不能吃什么、能吃什么（食物）、饮食建议。例：痛风不能吃什么？糖尿病人能吃水果吗？胆固醇高饮食怎么调整？
- disease_fact: 核心意图是了解某病/某症状的医学知识——包括：病因、症状表现、并发症、预防、诊断判断（是不是某病）、治疗方案（某病怎么治/怎么治疗）。例：糖尿病病因？感冒有什么症状？心口疼是不是心梗？肺炎怎么治疗？
- general_medical_qa: 兜底，仅当以上4类都不匹配时使用——包括：情境处置（发烧了怎么办、孩子抽搐怎么办）、检查建议（需要做CT吗）、报告解读（血糖高代表什么）、综合管理建议（平时注意什么、复查注意什么）。

【严格判断顺序】department → medication → diet → disease_fact → general_medical_qa
【关键区分】提到具体药名/药物 = medication；提到具体食物/饮食 = diet；问某病知识或诊断 = disease_fact；其余 = general_medical_qa

规则：
1. 若问题涉及疾病、症状、药物、饮食、科室，启用 kg。
2. 绝大多数医疗问题都应启用 qa。
3. 若含个人健康数据（"我"、"我的"、"病历"、"报告"、"指标"等），sets needs_case_context=true。

输出 JSON（不加其他文字）：
{{"kg": true/false, "qa": true/false, "query_type": "...", "needs_case_context": true/false, "answer_style": "...", "reason": "一句中文解释"}}"""

_ROUTER_USER = """用户问题: {query}

JSON:"""

# ---------------------------------------------------------------------------
# 路由器
# ---------------------------------------------------------------------------


class QueryRouter:
    """混合路由器：LLM 优先（专用 DeepSeek），附带规则回退。

    用法::

        router = QueryRouter()
        decision = router.route("感冒了怎么办")

        # 强制规则模式:
        decision = router.route("...", use_llm=False)

        # 无 DeepSeek API Key → 自动仅用规则:
        router = QueryRouter()
        decision = router.route("...")
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 已废弃，保留仅用于向后兼容。LLM 路由始终使用专用 DeepSeek 客户端。
        """
        self.llm = llm_client

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def route(self, query: str, use_llm: bool = True) -> Dict:
        """将 *query* 分类并返回路由 metadata。

        返回字典，键为：use_kg、use_qa、reason、query_type、answer_style、needs_case_context。
        不再返回 execution_mode——执行模式统一由上层决定。
        """
        if use_llm and settings.deepseek_api_key:
            result = self._llm_route(query)
            if result is not None:
                self._enrich_route(result, query)
                return result

        route = self._rule_route(query)
        self._enrich_route(route, query)
        return route

    # ------------------------------------------------------------------
    # LLM 路由
    # ------------------------------------------------------------------

    def _llm_route(self, query: str) -> Optional[Dict]:
        """尝试基于 LLM 的路由。任何失败均返回 None。"""
        try:
            client = _get_router_client()
            response = client.chat.completions.create(
                model=settings.deepseek_intent_model,
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
        """解析 LLM JSON 输出，转换简写键名并验证。"""
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
            "use_qa": data.get("qa", data.get("use_qa", False)),
            "reason": data.get("reason", ""),
            "query_type": data.get("query_type", ""),
            "needs_case_context": bool(data.get("needs_case_context", False)),
            "answer_style": data.get("answer_style", ""),
        }

        if canonical["query_type"] not in QUERY_TYPES:
            logger.debug("LLM route invalid query_type: %s", canonical["query_type"])
            return None

        # 确保至少启用一个数据源
        if not (canonical["use_kg"] or canonical["use_qa"]):
            canonical["use_qa"] = True

        return canonical

    # ------------------------------------------------------------------
    # 规则路由（回退）
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_route(query: str) -> Dict:
        for keywords, qtype, kg, toyhom in _ROUTES:
            if any(kw in query for kw in keywords):
                reason = _build_reason(kg, toyhom, keywords, query)
                return {
                    "use_kg": kg,
                    "use_qa": toyhom,
                    "reason": reason,
                    "query_type": qtype,
                }

        return {
            "use_kg": False,
            "use_qa": True,
            "reason": "未匹配到特定规则，默认使用通用问答库",
            "query_type": _FALLBACK_QUERY_TYPE,
        }

    @staticmethod
    def _enrich_route(route: Dict, query: str) -> None:
        """补齐下游 prompt / 病例检索需要的路由字段。"""
        qtype = route.get("query_type") or _FALLBACK_QUERY_TYPE
        if qtype not in QUERY_TYPES:
            qtype = _FALLBACK_QUERY_TYPE
            route["query_type"] = qtype
        if not route.get("answer_style"):
            route["answer_style"] = _ANSWER_STYLE_BY_TYPE.get(qtype, "general_guidance")
        route["needs_case_context"] = bool(
            route.get("needs_case_context")
            or any(keyword in query for keyword in _CASE_CONTEXT_KEYWORDS)
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _build_reason(kg: bool, toyhom: bool, keywords: list[str], query: str) -> str:
    parts: list[str] = []
    matched = [kw for kw in keywords if kw in query]
    tag = "、".join(matched[:3])
    if kg:
        parts.append(f"命中知识图谱关键词「{tag}」→ 开启 neo4j_kg")
    if toyhom:
        parts.append(f"命中关键词「{tag}」→ 开启 qa")
    return "；".join(parts) if parts else f"关键词匹配: {tag}"
