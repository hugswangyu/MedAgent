"""Milvus client wrapper for medical QA vectors."""

from __future__ import annotations

from typing import Dict, List

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from pymilvus.exceptions import MilvusException

from config.settings import settings


VARCHAR_LIMITS = {
    "pk": 128,
    "department": 128,
    "title": 512,
    "question": 4096,
    "answer": 8192,
    "text": 16384,
    "source": 128,
}


class MilvusClientWrapper:
    def __init__(
        self,
        host: str = settings.milvus_host,
        port: int = settings.milvus_port,
        collection_name: str = settings.milvus_collection,
        alias: str = "default",
    ):
        self.host = host
        self.port = str(port)
        self.collection_name = collection_name
        self.alias = alias
        self.collection: Collection | None = None

    def connect(self) -> None:
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        print(f"Milvus connected: {self.host}:{self.port}")

    def create_collection(self, embedding_dim: int, recreate: bool = False) -> Collection:
        if recreate and utility.has_collection(self.collection_name, using=self.alias):
            utility.drop_collection(self.collection_name, using=self.alias)
            print(f"Dropped Milvus collection: {self.collection_name}")

        if utility.has_collection(self.collection_name, using=self.alias):
            self.collection = Collection(self.collection_name, using=self.alias)
            print(f"Using existing Milvus collection: {self.collection_name}")
            return self.collection

        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=VARCHAR_LIMITS["pk"],
            ),
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=VARCHAR_LIMITS["department"]),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=VARCHAR_LIMITS["title"]),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=VARCHAR_LIMITS["question"]),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=VARCHAR_LIMITS["answer"]),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=VARCHAR_LIMITS["text"]),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=VARCHAR_LIMITS["source"]),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        schema = CollectionSchema(fields=fields, description="Toyhom medical QA vector collection")
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.alias,
            shards_num=2,
        )

        self.collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        print(f"Created Milvus collection: {self.collection_name}, dim={embedding_dim}")
        return self.collection

    def insert_batch(self, docs: List[Dict], embeddings: List[List[float]]) -> bool:
        if self.collection is None:
            self.collection = Collection(self.collection_name, using=self.alias)

        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must have the same length")
        if not docs:
            return True

        rows = []
        for doc, embedding in zip(docs, embeddings):
            rows.append(
                {
                    "pk": self._clip(str(doc["id"]), "pk"),
                    "department": self._clip(doc.get("department", ""), "department"),
                    "title": self._clip(doc.get("title", ""), "title"),
                    "question": self._clip(doc.get("question", ""), "question"),
                    "answer": self._clip(doc.get("answer", ""), "answer"),
                    "text": self._clip(doc.get("text", ""), "text"),
                    "source": self._clip(doc.get("source", "toyhom"), "source"),
                    "embedding": embedding,
                }
            )

        try:
            self.collection.insert(rows)
            self.collection.flush()
            return True
        except MilvusException as exc:
            if hasattr(self.collection, "upsert"):
                try:
                    self.collection.upsert(rows)
                    self.collection.flush()
                    print(f"warning: duplicate primary keys were upserted: {exc}")
                    return True
                except MilvusException as upsert_exc:
                    print(f"warning: skip batch after Milvus upsert failed: {upsert_exc}")
                    return False

            print(f"warning: skip batch after Milvus insert failed: {exc}")
            return False

    def load_collection(self) -> Collection:
        if self.collection is None:
            self.collection = Collection(self.collection_name, using=self.alias)
        self.collection.load()
        print(f"Milvus collection loaded: {self.collection_name}")
        return self.collection

    @staticmethod
    def _clip(value: object, field_name: str) -> str:
        text = "" if value is None else str(value)
        return text[: VARCHAR_LIMITS[field_name]]

