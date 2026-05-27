"""提示词构建器：将检索结果组装为最终的 LLM 提示词。"""

from __future__ import annotations

from typing import Dict, List, Optional

from medrag.prompts import (
    MEDICAL_ANSWER_PROMPT,
    CONTEXT_CASE_HEADER,
    CONTEXT_KG_HEADER,
    CONTEXT_TOYHOM_HEADER,
    CONTEXT_EMPTY_NOTE,
)

MAX_PER_SOURCE = 5          # 每个来源在提示词中的最大结果数
MAX_RESULT_CHARS = 400      # 每条结果截断长度


class PromptBuilder:
    """将多源检索结果组装为完整的答案生成提示词。

    用法::

        builder = PromptBuilder()
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            kg_results=kg_results,
            toyhom_results=toyhom_results,
            case_context=None,
            route=route,
        )
        # → 将 *prompt* 喂给 DeepSeek / OpenAI
    """

    def build_answer_prompt(
        self,
        query: str,
        kg_results: Optional[List[Dict]] = None,
        toyhom_results: Optional[List[Dict]] = None,
        case_context: Optional[str] = None,
        route: Optional[Dict] = None,
    ) -> str:
        """构建用于回答 LLM 的最终提示词字符串。

        Args:
            query: 用户原始问题。
            kg_results: KGRetriever.search() 输出（已重排序）。
            toyhom_results: ToyhomQARetriever.search() 输出（已重排序）。
            case_context: 预先计算的用户病例摘要，或 None。
            route: 路由器决策字典（目前未用，预留给未来基于 query_type 的提示词适配）。
        """
        # --- 组装上下文块 ---
        sections: list[str] = []

        # 1. 病例上下文（最高优先级）
        if case_context:
            sections.append(
                CONTEXT_CASE_HEADER.format(case_text=case_context.strip())
            )
        else:
            sections.append(
                CONTEXT_CASE_HEADER.format(case_text=CONTEXT_EMPTY_NOTE)
            )

        # 2. 知识图谱结果
        if kg_results:
            kg_text = self._format_kg_results(kg_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_KG_HEADER.format(kg_text=kg_text))
        else:
            sections.append(CONTEXT_KG_HEADER.format(kg_text=CONTEXT_EMPTY_NOTE))

        # 3. Toyhom 问答结果
        if toyhom_results:
            qa_text = self._format_toyhom_results(toyhom_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_TOYHOM_HEADER.format(qa_text=qa_text))
        else:
            sections.append(CONTEXT_TOYHOM_HEADER.format(qa_text=CONTEXT_EMPTY_NOTE))

        # --- 最终提示词 ---
        context = "\n".join(sections)
        return (
            MEDICAL_ANSWER_PROMPT
            + context
            + f"\n\n## 用户当前问题\n{query}\n\n请根据以上资料回答用户的问题。"
        )

    # ------------------------------------------------------------------
    # 格式化辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_kg_results(results: list[Dict]) -> str:
        """格式化知识图谱结果。"""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            intent = r.get("intent", "")
            answer = r.get("answer", "")
            if len(answer) > MAX_RESULT_CHARS:
                answer = answer[:MAX_RESULT_CHARS] + "…"
            lines.append(f"[{i}] ({intent}) {answer}")
        return "\n".join(lines)

    @staticmethod
    def _format_toyhom_results(results: list[Dict]) -> str:
        """格式化 Toyhom 问答结果。"""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            answer = r.get("answer", "")
            text = answer or title or ""
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + "…"
            department = r.get("department", "")
            prefix = f"[{i}] "
            if department:
                prefix += f"科室：{department} | "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)
