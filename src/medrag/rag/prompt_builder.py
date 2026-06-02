"""提示词构建器：将检索结果组装为最终的 LLM 提示词。"""

from __future__ import annotations

from typing import Dict, List, Optional

from medrag.prompts import (
    MEDICAL_ANSWER_PROMPT,
    NO_RETRIEVAL_ANSWER_PROMPT,
    QUERY_TYPE_HINTS,
    ANSWER_STYLE_HINTS,
    CONTEXT_CASE_HEADER,
    CONTEXT_USER_CASE_CHUNKS_HEADER,
    CONTEXT_KG_HEADER,
    CONTEXT_TOYHOM_HEADER,
    CONTEXT_EMPTY_NOTE,
)

MAX_PER_SOURCE = 5          # 每个来源在提示词中的最大结果数
MAX_RESULT_CHARS = 400      # 每条结果截断长度


class PromptBuilder:
    """将多源检索结果组装为完整的答案生成提示词。

    两层设计：
    - Tier 1（检索完整、一致、高置信）：基于检索回答 + 标注来源 + 结尾"请咨询医生确认"
    - Tier 2（其他所有情况）：统一回答"未找到确切信息，建议咨询专业医生"，不提供任何医学内容
    """

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tier1(retrieval_quality: Optional[Dict]) -> bool:
        """Tier 1 条件：检索到资料且置信度为 high。

        未传入 retrieval_quality 时默认走 Tier 1（向后兼容）。
        生产中 chat_service 始终会传入，主动触发 Tier 2 降级。
        """
        if not retrieval_quality:
            return True
        has_data = retrieval_quality.get("has_kg") or retrieval_quality.get("has_qa")
        confidence = retrieval_quality.get("confidence", "high")
        return bool(has_data) and confidence == "high"

    def _build_prompt_parts(
        self,
        query: str,
        kg_results: Optional[List[Dict]] = None,
        toyhom_results: Optional[List[Dict]] = None,
        case_results: Optional[List[Dict]] = None,
        case_context: Optional[str] = None,
        route: Optional[Dict] = None,
        retrieval_quality: Optional[Dict] = None,
        query_info: Optional[Dict] = None,
    ) -> tuple[str, str]:
        """返回 (system_part, user_part) 两个字符串。"""
        if self._is_tier1(retrieval_quality):
            return self._build_tier1_parts(
                query=query,
                kg_results=kg_results,
                toyhom_results=toyhom_results,
                case_results=case_results,
                case_context=case_context,
                route=route,
                query_info=query_info,
            )
        return self._build_tier2_parts(query)

    def build_answer_prompt(self, *args, **kwargs) -> str:
        """构建完整提示词字符串（向后兼容）。"""
        system, user = self._build_prompt_parts(*args, **kwargs)
        return system + user

    def build_messages(self, *args, **kwargs) -> list[dict]:
        """返回结构化的消息列表，供 LLM 调用时使用 system/user 角色分离。"""
        system, user = self._build_prompt_parts(*args, **kwargs)
        if system:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    # Tier 1：完整回答构建
    # ------------------------------------------------------------------

    def _build_tier1_parts(
        self,
        query: str,
        kg_results: Optional[List[Dict]] = None,
        toyhom_results: Optional[List[Dict]] = None,
        case_results: Optional[List[Dict]] = None,
        case_context: Optional[str] = None,
        route: Optional[Dict] = None,
        query_info: Optional[Dict] = None,
    ) -> tuple[str, str]:
        """构建 Tier 1 回答：基于检索资料 + 标注来源 + 结尾提示。"""
        # --- 根据 query_type 注入分类指令 ---
        query_type = (
            route.get("query_type", "general_medical_qa")
            if route else "general_medical_qa"
        )
        type_hint = QUERY_TYPE_HINTS.get(query_type, QUERY_TYPE_HINTS["general_medical_qa"])
        answer_style = (route or {}).get("answer_style", "general_guidance")
        if case_context or case_results:
            answer_style = "case_based"
        style_hint = ANSWER_STYLE_HINTS.get(answer_style, ANSWER_STYLE_HINTS["general_guidance"])
        system = (
            MEDICAL_ANSWER_PROMPT
            + "\n\n## 当前问题类型\n"
            + type_hint
            + "\n\n## 当前回答风格\n"
            + style_hint
        )

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

        if case_results:
            case_chunks = self._format_case_results(case_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_USER_CASE_CHUNKS_HEADER.format(case_chunks=case_chunks))
        else:
            sections.append(CONTEXT_USER_CASE_CHUNKS_HEADER.format(case_chunks=CONTEXT_EMPTY_NOTE))

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

        context = "\n".join(sections)
        query_block = self._format_query_block(query, query_info)
        user = (
            context
            + query_block
            + "\n\n请严格根据以上检索资料回答用户的问题，"
            "并在每个医学结论后标注来源。"
        )
        return system, user

    # ------------------------------------------------------------------
    # Tier 2：无检索结果的安全回答
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tier2_parts(query: str) -> tuple[str, str]:
        """构建 Tier 2 安全回答：直接输出固定话术，不提供任何医学内容。"""
        return ("", NO_RETRIEVAL_ANSWER_PROMPT.format(query=query))

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
            lines.append(f"[KG-{i}] ({intent}) {answer}")
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
            prefix = f"[QA-{i}] "
            if department:
                prefix += f"科室：{department} | "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)

    @staticmethod
    def _format_case_results(results: list[Dict]) -> str:
        """格式化当前用户病例片段。"""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            filename = r.get("filename", "")
            text = r.get("answer") or r.get("text") or ""
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + "…"
            prefix = f"[CASE-{i}] "
            if filename:
                prefix += f"文件：{filename} | "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)

    @staticmethod
    def _format_query_block(query: str, query_info: Optional[Dict]) -> str:
        if not query_info:
            return f"\n\n## 用户当前问题\n{query}"
        normalized = query_info.get("normalized_query", query)
        reason = query_info.get("rewrite_reason", "")
        terms = "、".join(query_info.get("medical_terms") or [])
        return (
            "\n\n## 用户当前问题\n"
            f"原始问题：{query}\n"
            f"检索规范化问题：{normalized}\n"
            f"规范化说明：{reason}"
            + (f"\n医学术语：{terms}" if terms else "")
        )
