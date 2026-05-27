"""Unified medical chat service: one entry point for the full RAG pipeline.

Orchestrates: retrieval → reranking → prompt building → generation → safety.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from medrag.config.settings import settings
from medrag.rag import PromptBuilder, AnswerGenerator, SafetyGuard
from medrag.retrieval import (
    HybridRetriever,
    KGRetriever,
    QueryRouter,
    get_reranker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MedicalChatService
# ---------------------------------------------------------------------------

# Default NER checkpoint (mirrors webui.load_model)
_DEFAULT_NER_CHECKPOINT = "best_roberta_rnn_model_ent_aug.pt"


class MedicalChatService:
    """End-to-end medical QA pipeline.

    Usage::

        service = MedicalChatService()
        result = service.chat("感冒了怎么办")

    All sub-components can be injected for testing or custom configuration::

        service = MedicalChatService(
            kg_retriever=my_kg,
            answer_generator=my_gen,
        )
    """

    def __init__(
        self,
        kg_retriever=None,          # KGRetriever instance or None → auto-load
        toyhom_retriever=None,      # ToyhomQARetriever or None → auto-create
        router=None,                # QueryRouter or None → auto-create
        hybrid_retriever=None,      # HybridRetriever or None → assemble from above
        reranker=None,              # reranker instance or None → get_reranker()
        prompt_builder=None,        # PromptBuilder or None → auto-create
        answer_generator=None,      # AnswerGenerator or None → auto-create
        safety_guard=None,          # SafetyGuard or None → auto-create
        ner_checkpoint: str = _DEFAULT_NER_CHECKPOINT,
    ):
        # ---- Retrieval pipeline ----
        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        else:
            _kg = kg_retriever or self._load_kg_retriever(ner_checkpoint)

            from medrag.vectors.toyhom_retriever import ToyhomQARetriever
            _toyhom = toyhom_retriever or ToyhomQARetriever()

            _router = router or QueryRouter()

            self.hybrid_retriever = HybridRetriever(
                kg_retriever=_kg,
                toyhom_retriever=_toyhom,
                router=_router,
            )

        # ---- Generation pipeline ----
        self.reranker = reranker or get_reranker()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.safety_guard = safety_guard or SafetyGuard()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        user_case_summary: Optional[str] = None,
    ) -> Dict:
        """Run the full medical QA pipeline.

        Args:
            query: The user's medical question.
            user_case_summary: Pre-computed case summary from uploaded
                               records, or ``None``.

        Returns:
            Dict with keys: ``answer``, ``route``, ``kg_results``,
            ``toyhom_results``, ``reranked_results``, ``risk_info``.
        """
        # 1. Risk detection (before retrieval, so we can flag early)
        risk_info = self.safety_guard.detect_risk(query)

        # 2. Multi-source retrieval
        retrieval = self.hybrid_retriever.retrieve(query)

        # 3. Rerank
        reranked = self.reranker.rerank(
            query,
            retrieval["all_results"],
            top_k=settings.rerank_top_k,
        )

        # 4. Build prompt (case context is injected directly — always
        #    included when available, bypassing the Router's decision)
        prompt = self.prompt_builder.build_answer_prompt(
            query=query,
            kg_results=retrieval["kg_results"],
            toyhom_results=retrieval["toyhom_results"],
            case_context=user_case_summary,
            route=retrieval["route"],
        )

        # 5. Generate answer
        answer = self.answer_generator.generate(prompt)

        # 6. Inject safety notices
        answer = self.safety_guard.append_safety_notice(answer, risk_info)

        return {
            "answer": answer,
            "route": retrieval["route"],
            "kg_results": retrieval["kg_results"],
            "toyhom_results": retrieval["toyhom_results"],
            "reranked_results": reranked,
            "risk_info": risk_info,
        }

    # ------------------------------------------------------------------
    # Internal: NER model loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_kg_retriever(checkpoint: str) -> KGRetriever:
        """Load NER model and build a KGRetriever.

        Mirrors the ``load_model()`` path in ``webui.py``.  If the
        checkpoint file is missing, KG search will be unavailable.
        """
        import pickle

        import torch
        from transformers import BertTokenizer

        from medrag.ner import model as zwk

        with open(PROJECT_ROOT / "tmp_data" / "tag2idx.npy", "rb") as f:
            tag2idx = pickle.load(f)
        idx2tag = list(tag2idx)

        rule = zwk.rule_find()
        tfidf_r = zwk.tfidf_alignment()

        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        bert = zwk.Bert_Model(
            "hfl/chinese-roberta-wwm-ext",
            hidden_size=128,
            tag_num=len(tag2idx),
            bi=True,
        )

        checkpoint_path = PROJECT_ROOT / "model" / checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        bert = bert.to(device)
        bert.eval()

        logger.info("KGRetriever loaded (NER checkpoint=%s, device=%s)", checkpoint, device)

        return KGRetriever(
            bert_model=bert,
            bert_tokenizer=tokenizer,
            rule=rule,
            tfidf_r=tfidf_r,
            device=device,
            idx2tag=idx2tag,
        )
