"""Prompt builder: assembles retrieval results into a final LLM prompt."""

from __future__ import annotations

from typing import Dict, List, Optional

from medrag.config.prompt_templates import (
    MEDICAL_ANSWER_PROMPT,
    CONTEXT_CASE_HEADER,
    CONTEXT_KG_HEADER,
    CONTEXT_TOYHOM_HEADER,
    CONTEXT_EMPTY_NOTE,
)

MAX_PER_SOURCE = 5          # max results per source in the prompt
MAX_RESULT_CHARS = 400      # truncate each result to this length


class PromptBuilder:
    """Assemble full answer-generation prompt from multi-source results.

    Usage::

        builder = PromptBuilder()
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            kg_results=kg_results,
            toyhom_results=toyhom_results,
            case_context=None,
            route=route,
        )
        # → feed *prompt* to DeepSeek / OpenAI
    """

    def build_answer_prompt(
        self,
        query: str,
        kg_results: Optional[List[Dict]] = None,
        toyhom_results: Optional[List[Dict]] = None,
        case_context: Optional[str] = None,
        route: Optional[Dict] = None,
    ) -> str:
        """Build the final prompt string for the answering LLM.

        Args:
            query: Original user question.
            kg_results: KGRetriever.search() output (already re-ranked).
            toyhom_results: ToyhomQARetriever.search() output (already re-ranked).
            case_context: Pre-computed user case summary, or None.
            route: Router decision dict (unused for now; reserved for
                   future prompt adaptation based on query_type).
        """
        # --- Assemble context blocks ---
        sections: list[str] = []

        # 1. Case context (highest priority)
        if case_context:
            sections.append(
                CONTEXT_CASE_HEADER.format(case_text=case_context.strip())
            )
        else:
            sections.append(
                CONTEXT_CASE_HEADER.format(case_text=CONTEXT_EMPTY_NOTE)
            )

        # 2. KG results
        if kg_results:
            kg_text = self._format_kg_results(kg_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_KG_HEADER.format(kg_text=kg_text))
        else:
            sections.append(CONTEXT_KG_HEADER.format(kg_text=CONTEXT_EMPTY_NOTE))

        # 3. Toyhom QA results
        if toyhom_results:
            qa_text = self._format_toyhom_results(toyhom_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_TOYHOM_HEADER.format(qa_text=qa_text))
        else:
            sections.append(CONTEXT_TOYHOM_HEADER.format(qa_text=CONTEXT_EMPTY_NOTE))

        # --- Final prompt ---
        context = "\n".join(sections)
        return (
            MEDICAL_ANSWER_PROMPT
            + context
            + f"\n\n## 用户当前问题\n{query}\n\n请根据以上资料回答用户的问题。"
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_kg_results(results: list[Dict]) -> str:
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
