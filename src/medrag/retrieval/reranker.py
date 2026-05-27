"""Rerankers for multi-source retrieval results.

Two strategies, sharing the same ``rerank(query, results, top_k)``
interface:

  - CrossEncoderReranker  neural cross-encoder (production default)
  - SimpleReranker        rule-based (zero-latency fallback)

Use ``get_reranker()`` to auto-select the best available one.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SOURCE_BONUS: Dict[str, float] = {
    "neo4j_kg": 0.20,
    "toyhom_qa": 0.0,
    "user_case": 0.30,
}
DEFAULT_SCORE = 0.50


def _result_text(result: Dict) -> str:
    """Combine all text-like fields in a result for matching / cross-encoding."""
    parts: list[str] = []
    for key in ("answer", "text", "title", "question"):
        v = result.get(key, "")
        if v:
            parts.append(str(v))
    evidence = result.get("evidence")
    if isinstance(evidence, list):
        parts.append(" ".join(str(x) for x in evidence))
    elif evidence:
        parts.append(str(evidence))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 1. SimpleReranker  (rule-based)
# ---------------------------------------------------------------------------

KEYWORD_HIT_BONUS = 0.10
MAX_KEYWORD_BONUS = 0.30


class SimpleReranker:
    """Rule-based: source prior + keyword overlap."""

    def rerank(self, query: str, results: List[Dict],
               top_k: int = 8) -> List[Dict]:
        if not results:
            return []
        keywords = _extract_keywords(query)
        for r in results:
            base = SOURCE_BONUS.get(r.get("source", ""), 0.0) + r.get("score", DEFAULT_SCORE)
            kw_hits = sum(1 for kw in keywords if kw in _result_text(r)) if keywords else 0
            kw_bonus = min(kw_hits * KEYWORD_HIT_BONUS, MAX_KEYWORD_BONUS)
            r["final_score"] = round(base + kw_bonus, 4)
            r["rerank_reason"] = (
                f"source={r.get('source')}, base={base:.3f}"
                + (f", kw×{kw_hits} +{kw_bonus:.2f}" if kw_bonus else "")
            )
        return sorted(results, key=lambda r: r["final_score"], reverse=True)[:top_k]


def _extract_keywords(query: str) -> List[str]:
    q = query.replace("？", "").replace("?", "").replace("，", "").replace("。", "")
    if len(q) <= 2:
        return [q]
    kw: list[str] = []
    for n in (2, 3, 4):
        if len(q) >= n:
            kw.extend(q[i:i + n] for i in range(len(q) - n + 1))
    return kw


# ---------------------------------------------------------------------------
# 2. CrossEncoderReranker  (neural)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Neural cross-encoder for relevance scoring.

    Uses ``sentence-transformers`` CrossEncoder.  The model is downloaded
    on first instantiation (~1 GB for ``BAAI/bge-reranker-base``).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None  # lazy load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: List[Dict],
               top_k: int = 8) -> List[Dict]:
        if not results:
            return []
        pairs = [(query, _result_text(r)) for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for r, ce_score in zip(results, scores):
            src_bonus = SOURCE_BONUS.get(r.get("source", ""), 0.0)
            r["final_score"] = round(float(ce_score) + src_bonus, 4)
            r["rerank_reason"] = (
                f"source={r.get('source')}, ce={ce_score:.4f}"
                + (f" +src_bonus={src_bonus:.2f}" if src_bonus else "")
            )

        return sorted(results, key=lambda r: r["final_score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Factory: best-available reranker with fallback chain
# ---------------------------------------------------------------------------

_DEFAULT_CROSS_MODEL = "BAAI/bge-reranker-base"


def get_reranker(cross_model: str = _DEFAULT_CROSS_MODEL):
    """Return the best available reranker.

    Priority: CrossEncoder → SimpleReranker (zero-dep fallback).

    Usage::

        reranker = get_reranker()
        ranked = reranker.rerank(query, results, top_k=8)
    """
    try:
        cross = CrossEncoderReranker(model_name=cross_model)
        _ = cross.model  # trigger download / validation
        logger.info("Using CrossEncoderReranker (model=%s)", cross_model)
        return cross
    except Exception:
        logger.warning(
            "CrossEncoder unavailable, falling back to SimpleReranker",
            exc_info=True,
        )
        return SimpleReranker()
