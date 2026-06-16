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
    ):
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._top_k = top_k
        self._max_results = max_results

    def execute(self, query: str) -> str:
        """执行完整 RAG pipeline，返回格式化文本。

        Args:
            query: 检索关键词。

        Returns:
            格式化文本，包含知识图谱结果和相似问答结果。
            空结果时返回 "未找到相关信息。"。
        """
        try:
            retrieval = self._retriever.retrieve(query)
        except Exception as exc:
            return f"检索出错：{exc}"

        kg_results = retrieval.get("kg_results", [])
        qa_results = retrieval.get("qa_results", [])

        # 重排 QA 结果
        if qa_results:
            try:
                qa_results = self._reranker.rerank(query, qa_results, top_k=self._top_k)
            except Exception:
                qa_results = qa_results[:self._top_k]

        # 缓存检索结果供 stream_chat() trace 使用，避免重复检索
        self._retriever._last_raw_result = retrieval
        self._retriever._last_reranked_qa = qa_results

        parts: List[str] = []

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

        return "\n".join(parts).strip() if parts else "未找到相关信息。"
