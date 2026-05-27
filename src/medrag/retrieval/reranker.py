"""多源检索结果重排序器。

两种策略，共享相同的 ``rerank(query, results, top_k)`` 接口：

  - CrossEncoderReranker  神经网络交叉编码器（生产环境默认）
  - SimpleReranker        规则评分（零延迟回退方案）

使用 ``get_reranker()`` 自动选择最佳可用方案。
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 共享辅助函数
# ---------------------------------------------------------------------------

SOURCE_BONUS: Dict[str, float] = {
    "neo4j_kg": 0.20,
    "toyhom_qa": 0.0,
    "user_case": 0.30,
}
DEFAULT_SCORE = 0.50


def _result_text(result: Dict) -> str:
    """组合结果中所有文本类字段，用于匹配 / 交叉编码。"""
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
# 1. SimpleReranker（规则评分）
# ---------------------------------------------------------------------------

KEYWORD_HIT_BONUS = 0.10
MAX_KEYWORD_BONUS = 0.30


class SimpleReranker:
    """规则评分：来源先验加权 + 关键词重叠。"""

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
# 2. CrossEncoderReranker（神经网络）
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """神经网络交叉编码器，用于相关性评分。

    使用 ``sentence-transformers`` CrossEncoder。模型在首次实例化时
    下载（``BAAI/bge-reranker-base`` 约 1 GB）。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None  # 延迟加载

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
# 工厂函数：自动选择最佳可用的重排序器（含回退链）
# ---------------------------------------------------------------------------

_DEFAULT_CROSS_MODEL = "BAAI/bge-reranker-base"


def get_reranker(cross_model: str = _DEFAULT_CROSS_MODEL):
    """返回最佳可用的重排序器。

    优先级：CrossEncoder → SimpleReranker（零依赖回退）。

    用法::

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
