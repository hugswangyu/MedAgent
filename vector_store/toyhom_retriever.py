"""Toyhom medical QA retriever backed by Milvus / Zilliz Cloud."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import Collection

from config.settings import settings
from vector_store.embedding import EmbeddingModel
from vector_store.milvus_client import MilvusClientWrapper


class ToyhomQARetriever:
    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
    ):
        self.embedding_model = EmbeddingModel(model_name)

        milvus = MilvusClientWrapper()
        milvus.connect()
        self.collection = milvus.load_collection()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        department: Optional[str] = None,
    ) -> List[Dict]:
        if top_k is None:
            top_k = settings.retrieval_top_k
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
                    "source": "toyhom_qa",
                    "id": hit.entity.get("pk"),
                    "score": float(hit.distance),
                    "department": hit.entity.get("department"),
                    "title": hit.entity.get("title"),
                    "question": hit.entity.get("question"),
                    "answer": hit.entity.get("answer"),
                    "text": hit.entity.get("text"),
                }
            )
        return hits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Toyhom medical QA via Milvus.")
    parser.add_argument("query", nargs="?", help="Search query.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results.")
    parser.add_argument("--department", default=None, help="Filter by department.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query = args.query
    if not query:
        query = input("请输入问题: ").strip()
        if not query:
            print("查询不能为空")
            sys.exit(1)

    retriever = ToyhomQARetriever()
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
