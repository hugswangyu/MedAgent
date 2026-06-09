"""医疗问答检索器，基于 Milvus / Zilliz Cloud（cMedQA2 数据集）。"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional

from pymilvus import Collection

from medrag.config.settings import settings
from medrag.vectors.embedding import EmbeddingModel
from medrag.vectors.milvus_client import MilvusClientWrapper

logger = logging.getLogger(__name__)


class QARetriever:
    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
    ):
        self.embedding_model = None
        self.collection = None
        self._available = False

        try:
            self.embedding_model = EmbeddingModel(model_name)
        except Exception as exc:
            logger.warning("EmbeddingModel init failed: %s", exc)

        try:
            milvus = MilvusClientWrapper()
            milvus.connect()
            self.collection = milvus.load_collection()
            self._available = True
        except Exception as exc:
            logger.warning("Milvus connection failed: %s", exc)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        department: Optional[str] = None,
    ) -> List[Dict]:
        if not self._available or self.collection is None:
            raise RuntimeError("QARetriever unavailable (Milvus or embedding not loaded)")
        if top_k is None:
            top_k = settings.retrieval_top_k
        if self.embedding_model is None:
            raise RuntimeError("QARetriever unavailable (embedding model not loaded)")
        query_embedding = self.embedding_model.encode_one(query, is_query=True)

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 64},
        }

        expr: Optional[str] = None
        if department:
            expr = f'department == "{department}"'

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "pk",
                "department",
                "title",
                "question",
                "answer",
                "text",
            ],
        )

        hits: List[Dict] = []
        for hit in results[0]:
            hits.append(
                {
                    "source": "cmedqa2",
                    "id": hit.entity.get("pk"),
                    "score": float(hit.distance),
                    "department": hit.entity.get("department") or "",
                    "title": hit.entity.get("title") or "",
                    "question": hit.entity.get("question") or "",
                    "answer": hit.entity.get("answer") or "",
                    "text": hit.entity.get("text") or "",
                }
            )
        return hits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通过 Milvus 搜索医疗问答（cMedQA2）。")
    parser.add_argument("query", nargs="?", help="搜索查询。")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量。")
    parser.add_argument("--department", default=None, help="按科室过滤。")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query = args.query
    if not query:
        query = input("请输入问题: ").strip()
        if not query:
            print("查询不能为空")
            sys.exit(1)

    retriever = QARetriever()
    results = retriever.search(query, top_k=args.top_k, department=args.department)

    print(f"\n查询: {query}")
    if args.department:
        print(f"科室过滤: {args.department}")
    print(f"共找到 {len(results)} 条结果:\n")

    for i, r in enumerate(results, 1):
        print(f"--- Top {i} (score={r['score']:.4f}, dept={r['department']}) ---")
        print(f"  标题: {r['title']}")
        print(f"  问题: {r['question']}")
        print(f"  回答: {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}")
        print()


if __name__ == "__main__":
    main()
