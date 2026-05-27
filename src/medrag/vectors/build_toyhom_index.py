"""为 Toyhom 医学问答数据集构建 Milvus 向量索引。"""

from __future__ import annotations

import argparse
from typing import Iterable, List

from medrag.config.settings import settings
from medrag.data.toyhom_loader import load_toyhom_dataset


def _batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_toyhom_index(
    data_root: str | Path = settings.toyhom_dataset_path,
    batch_size: int = 128,
    limit: int | None = 10000,
    recreate: bool = False,
) -> None:
    if limit == 0:
        limit = None
    docs = load_toyhom_dataset(data_root, limit=limit)
    print(f"Loaded Toyhom docs: {len(docs)}")
    if not docs:
        return

    from medrag.vectors.embedding import EmbeddingModel
    from medrag.vectors.milvus_client import MilvusClientWrapper

    embedding_model = EmbeddingModel(settings.embedding_model_name)
    milvus = MilvusClientWrapper()
    milvus.connect()
    milvus.create_collection(embedding_model.embedding_dim, recreate=recreate)

    inserted = 0
    skipped = 0
    total = len(docs)
    for batch_docs in _batched(docs, batch_size):
        texts = [doc["title"] for doc in batch_docs]
        embeddings = embedding_model.encode(texts, batch_size=batch_size, is_query=False)
        ok = milvus.insert_batch(batch_docs, embeddings)
        if ok:
            inserted += len(batch_docs)
        else:
            skipped += len(batch_docs)
        print(f"Indexed {inserted}/{total}, skipped={skipped}")

    milvus.flush()
    milvus.create_index()
    milvus.load_collection()
    print(f"Toyhom index build finished. inserted={inserted}, skipped={skipped}, total={total}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 Milvus 中构建 Toyhom 医学问答向量索引。")
    parser.add_argument("--data_root", default=str(settings.toyhom_dataset_path), help="Toyhom 数据集根目录。")
    parser.add_argument("--batch_size", type=int, default=128, help="嵌入与插入的批量大小。")
    parser.add_argument("--limit", type=int, default=10000, help="最大索引文档数。传 0 表示无限制。")
    parser.add_argument("--recreate", action="store_true", help="删除并重新创建集合。")
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
