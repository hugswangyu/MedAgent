"""src/medrag/react/rag_tool.py — 将 RAG pipeline 包装为 ReAct 工具。

``RetrieveKnowledgeTool`` 是一个 ReAct 工具，内部走完整检索-重排-组装
流水线，返回格式化文本供 LLM 推理使用。这样 RAG 变成 ReAct 工具集中
的一个选项，LLM 自主决定何时需要检索知识。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class RetrieveKnowledgeTool:
    """ReAct 工具：检索医学知识。

    内部调用 hybrid_retriever.retrieve() + reranker.rerank()，
    将结果格式化为文本返回给 ReAct 循环的 LLM。
    """

    name = "retrieve_knowledge"

    # 工具描述会直接出现在 LLM 提示词中，精确的描述帮助 LLM 决定何时调用
    description = (
        "检索医学知识库获取疾病、症状、药物、检查等医学信息。"
        "当你需要医学事实信息来回答问题时调用此工具。"
        "会同时搜索知识图谱和医疗问答库。"
    )

    parameters: List[Dict] = [
        {
            "name": "query",
            "type": "string",
            "description": "要检索的医学问题或关键词",
        },
    ]

    def __init__(
        self,
        hybrid_retriever: Any,
        reranker: Any,
        prompt_builder: Any = None,
        top_k: int = 5,
        max_results: int = 5,
        username: Optional[str] = None,
        department: Optional[str] = None,
        route: Optional[Dict] = None,
        trace_context: Optional[Dict] = None,
    ):
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._top_k = top_k
        self._max_results = max_results
        self._username = username
        self._department = department
        self._route = route
        self._trace_context = trace_context

    def execute(self, query: str) -> str:
        """执行完整 RAG pipeline，返回格式化文本。

        Args:
            query: 检索关键词。

        Returns:
            格式化文本，包含知识图谱结果和相似问答结果。
            空结果时返回 "未找到相关信息。"。
        """
        try:
            retrieval = self._retriever.retrieve(
                query,
                department=self._department,
                username=self._username,
                route=self._route,
            )
        except Exception as exc:
            return f'[status=error] 检索系统异常：{exc}。请勿重试此工具，直接告知用户无法获取信息。'

        kg_results = retrieval.get("kg_results", [])
        qa_results = retrieval.get("qa_results", [])
        case_results = retrieval.get("case_results", [])

        # 重排 QA 结果
        if qa_results:
            try:
                qa_results = self._reranker.rerank(query, qa_results, top_k=self._top_k)
            except Exception:
                qa_results = qa_results[:self._top_k]

        # 请求级 trace 容器，避免把临时结果写入共享 Retriever。
        if self._trace_context is not None:
            self._trace_context["raw_result"] = retrieval
            self._trace_context["reranked_qa"] = qa_results

        parts: List[str] = []

        # 复用统一的 RAG 章节构建规则。这里返回的是 Observation，
        # 历史内容只作为事实资料，不作为可执行指令。
        if self._prompt_builder is not None:
            try:
                sections = self._prompt_builder.build_sections(
                    query=query,
                    kg_results=kg_results,
                    qa_results=qa_results,
                    case_results=case_results,
                    route=retrieval.get("route"),
                    query_info=retrieval.get("query_info"),
                )
                parts = [
                    sections[key]
                    for key in ("case_chunks", "kg", "qa")
                    if sections.get(key)
                ]
            except Exception:
                parts = []

        if not parts:
            if case_results:
                parts.append("【用户病例片段】")
                for r in case_results[:self._max_results]:
                    content = r.get("answer") or r.get("text") or str(r)
                    parts.append(f"- {content[:500]}")
                parts.append("")

            if kg_results:
                parts.append("【知识图谱结果】")
                for r in kg_results[:self._max_results]:
                    content = r.get("answer") or r.get("text") or str(r)
                    parts.append(f"- {content[:500]}")
                parts.append("")

            if qa_results:
                parts.append("【相似问答结果】")
                for r in qa_results[:self._max_results]:
                    content = r.get("answer") or r.get("text") or str(r)
                    parts.append(f"- {content[:500]}")
                parts.append("")

        if not parts:
            return "[status=not_found] 检索完成，但知识库中无匹配文档。可尝试换用不同关键词，或直接告知用户信息不足。"
        return "\n".join(parts).strip()
