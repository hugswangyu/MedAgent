"""Build a Milvus vector index for the Toyhom medical QA dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from data_processor.toyhom_loader import load_toyhom_dataset


def _batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_toyhom_index(
    data_root: str | Path = settings.toyhom_dataset_path,
    batch_size: int = 32,
    limit: int | None = None,
    recreate: bool = False,
) -> None:
    docs = load_toyhom_dataset(data_root, limit=limit)
    print(f"Loaded Toyhom docs: {len(docs)}")
    if not docs:
        return

    from vector_store.embedding import EmbeddingModel
    from vector_store.milvus_client import MilvusClientWrapper

    embedding_model = EmbeddingModel(settings.embedding_model_name)
    milvus = MilvusClientWrapper()
    milvus.connect()
    milvus.create_collection(embedding_model.embedding_dim, recreate=recreate)

    inserted = 0
    skipped = 0
    total = len(docs)
    for batch_docs in _batched(docs, batch_size):
        texts = [doc["text"] for doc in batch_docs]
        embeddings = embedding_model.encode(texts, batch_size=batch_size, is_query=False)
        ok = milvus.insert_batch(batch_docs, embeddings)
        if ok:
            inserted += len(batch_docs)
        else:
            skipped += len(batch_docs)
        print(f"Indexed {inserted}/{total}, skipped={skipped}")

    milvus.load_collection()
    print(f"Toyhom index build finished. inserted={inserted}, skipped={skipped}, total={total}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Toyhom medical QA vectors in Milvus.")
    parser.add_argument("--data_root", default=str(settings.toyhom_dataset_path), help="Toyhom dataset root.")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding and insert batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum documents to index.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_toyhom_index(
        data_root=args.data_root,
        batch_size=args.batch_size,
        limit=args.limit,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
