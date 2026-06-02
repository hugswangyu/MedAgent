"""统一医疗对话服务：完整 RAG 流水线的单一入口。

编排：检索 → 重排序 → 提示词构建 → 生成 → 安全检查。
"""

from __future__ import annotations

import logging
from typing import Dict, Generator, Optional

from medrag.config.settings import settings
from medrag.llm import get_llm_client, get_llm_provider
from medrag.rag import PromptBuilder, AnswerGenerator, SafetyGuard
from medrag.retrieval import (
    HybridRetriever,
    KGRetriever,
    QueryNormalizer,
    QueryRouter,
    get_reranker,
)
from medrag.data.user_case_store import UserCaseRetriever, get_combined_case_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MedicalChatService
# ---------------------------------------------------------------------------


class MedicalChatService:
    """端到端医疗问答流水线。

    用法::

        service = MedicalChatService()
        result = service.chat("感冒了怎么办")

    所有子组件均可注入以便测试或自定义配置::

        service = MedicalChatService(
            kg_retriever=my_kg,
            answer_generator=my_gen,
        )
    """

    def __init__(
        self,
        kg_retriever=None,          # KGRetriever 实例 或 None → 自动加载
        toyhom_retriever=None,      # ToyhomQARetriever 或 None → 自动创建
        router=None,                # QueryRouter 或 None → 自动创建
        hybrid_retriever=None,      # HybridRetriever 或 None → 由上述组件组装
        reranker=None,              # reranker 实例 或 None → get_reranker()
        prompt_builder=None,        # PromptBuilder 或 None → 自动创建
        answer_generator=None,      # AnswerGenerator 或 None → 自动创建
        safety_guard=None,          # SafetyGuard 或 None → 自动创建
        case_retriever=None,        # UserCaseRetriever 或 None → 自动创建
        normalizer=None,            # QueryNormalizer 或 None → 自动创建
    ):
        # ---- 共享 LLM 客户端 ----
        _llm_client = get_llm_client()
        _llm_provider = get_llm_provider()

        # ---- 检索流水线 ----
        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        else:
            if kg_retriever is None:
                from pathlib import Path
                from medrag.infrastructure.ner import load_ner_model
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                kg_retriever = load_ner_model(project_root, llm_client=_llm_client)

            from medrag.vectors.toyhom_retriever import ToyhomQARetriever
            _toyhom = toyhom_retriever or ToyhomQARetriever()

            _router = router or QueryRouter(llm_client=_llm_client)

            self.hybrid_retriever = HybridRetriever(
                kg_retriever=kg_retriever,
                toyhom_retriever=_toyhom,
                router=_router,
                case_retriever=case_retriever or UserCaseRetriever(),
                normalizer=normalizer or QueryNormalizer(),
            )

        # ---- 生成流水线 ----
        self.reranker = reranker or get_reranker()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.answer_generator = answer_generator or AnswerGenerator(llm_provider=_llm_provider)
        self.safety_guard = safety_guard or SafetyGuard()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict:
        """运行完整的医疗问答流水线。

        Args:
            query: 用户的医学问题。
            user_case_summary: 预先计算的病例摘要，或 ``None``。

        Returns:
            字典，键为：``answer``、``route``、``kg_results``、
            ``toyhom_results``、``reranked_results``、``risk_info``。
        """
        # 1. 风险检测（在检索之前，以便尽早标记）
        risk_info = self.safety_guard.detect_risk(query)

        # 2. 多源检索
        retrieval = self.hybrid_retriever.retrieve(query, username=username)

        # 3. 重排序
        if user_case_summary is None and username:
            user_case_summary = get_combined_case_summary(username)

        messages = self.prompt_builder.build_messages(
            query=query,
            kg_results=retrieval["kg_results"],
            toyhom_results=retrieval["toyhom_results"],
            case_results=retrieval.get("case_results", []),
            case_context=user_case_summary,
            route=retrieval["route"],
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )

        # 6. 生成回答
        answer = self.answer_generator.generate(messages)

        # 7. 注入安全提示
        query_type = retrieval.get("route", {}).get("query_type")
        answer = self.safety_guard.append_safety_notice(
            answer, risk_info, retrieval_quality=retrieval_quality["confidence"],
            query_type=query_type,
        )

        return {
            "answer": answer,
            "route": retrieval["route"],
            "kg_results": retrieval["kg_results"],
            "toyhom_results": retrieval["toyhom_results"],
            "case_results": retrieval.get("case_results", []),
            "reranked_results": reranked,
            "risk_info": risk_info,
            "query_info": retrieval.get("query_info"),
        }

    def stream_chat(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
        username: Optional[str] = None,
        department: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """流式版本的医疗 QA 流水线。

        逐步 yield 事件字典：
        ``{"type": "rag_step", "step": {...}}``
        ``{"type": "trace", "rag_trace": {...}}``
        ``{"type": "content", "content": "token"}``

        Args:
            query: 用户问题。
            user_case_summary: 可选的病例摘要。
            department: 可选科室过滤（非"全科"时传入 ToyhomQARetriever）。
            provider: 可选 LLM 提供商（deepseek/zhipuai/ollama），默认使用 settings.llm_provider。
            model: 可选模型名，默认使用对应 provider 的默认模型。
        """
        # 动态创建 AnswerGenerator（仅在 provider 不同时切换）
        if provider and provider != settings.llm_provider:
            _gen_provider = get_llm_provider(provider)
            answer_gen = AnswerGenerator(llm_provider=_gen_provider)
        else:
            answer_gen = self.answer_generator
        # 1. 风险检测
        yield {
            "type": "rag_step",
            "step": {"key": "risk", "label": "安全检测", "icon": "🛡️", "detail": "扫描风险关键词"},
        }
        risk_info = self.safety_guard.detect_risk(query)

        # 2. 多源检索（按科室过滤）
        yield {
            "type": "rag_step",
            "step": {"key": "retrieve", "label": "多源检索", "icon": "🔍", "detail": "知识图谱 + 向量库"},
        }
        retrieval = self.hybrid_retriever.retrieve(query, department=department, username=username)

        # 3. 重排序
        yield {
            "type": "rag_step",
            "step": {"key": "rerank", "label": "结果重排序", "icon": "📊", "detail": f"共 {len(retrieval['all_results'])} 条候选"},
        }
        reranked = self.reranker.rerank(
            query,
            retrieval["all_results"],
            top_k=settings.rerank_top_k,
        )

        # 4. 计算检索质量
        retrieval_quality = {
            "has_kg": bool(retrieval["kg_results"]),
            "has_qa": bool(retrieval["toyhom_results"]),
            "has_case": bool(retrieval.get("case_results")),
            "confidence": (
                "high" if (retrieval["kg_results"] or retrieval["toyhom_results"] or retrieval.get("case_results"))
                else "none"
            ),
        }

        # 5. 构建提示词
        yield {
            "type": "rag_step",
            "step": {"key": "prompt", "label": "构建提示词", "icon": "📝"},
        }
        if user_case_summary is None and username:
            user_case_summary = get_combined_case_summary(username)

        messages = self.prompt_builder.build_messages(
            query=query,
            kg_results=retrieval["kg_results"],
            toyhom_results=retrieval["toyhom_results"],
            case_results=retrieval.get("case_results", []),
            case_context=user_case_summary,
            route=retrieval["route"],
            retrieval_quality=retrieval_quality,
            query_info=retrieval.get("query_info"),
        )

        # 5. 发送检索溯源信息
        rag_trace = {
            "tool_used": True,
            "tool_name": "multi-source-retrieval",
            "retrieval_stage": "initial",
            "retrieval_mode": retrieval["route"].get("query_type", ""),
            "query_info": retrieval.get("query_info"),
            "initial_retrieved_chunks": [
                {
                    "filename": r.get("source", r.get("id", "")),
                    "text": (r.get("answer") or r.get("text") or r.get("evidence", ""))[:200],
                    "rrf_rank": i + 1,
                    "rrf_score": r.get("rrf_score", 0),
                    "source_rank": r.get("rrf_source_rank", 0),
                }
                for i, r in enumerate(retrieval["all_results"][:10])
            ],
        }
        yield {"type": "trace", "rag_trace": rag_trace}

        # 6. LLM 流式生成
        yield {
            "type": "rag_step",
            "step": {"key": "generate", "label": "生成回答", "icon": "✨"},
        }
        full_answer = ""
        for token in answer_gen.generate_stream(messages, model=model):
            full_answer += token
            yield {"type": "content", "content": token}

        # 7. 安全提示尾注（作为最后一个 content 事件发送）
        query_type = retrieval.get("route", {}).get("query_type")
        footer = self.safety_guard.append_safety_notice(
            "", risk_info, retrieval_quality=retrieval_quality["confidence"],
            query_type=query_type,
        )
        if footer.strip():
            yield {"type": "content", "content": "\n\n" + footer}
