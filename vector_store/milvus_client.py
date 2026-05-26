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
    "question": 65535,
    "answer": 65535,
    "text": 65535,
    "source": 128,
}


class MilvusClientWrapper:
    def __init__(
        self,
        host: str = settings.milvus_host,
        port: int = settings.milvus_port,
        uri: str = settings.milvus_uri,
        token: str = settings.milvus_token,
        collection_name: str = settings.milvus_collection,
        alias: str = "default",
    ):
        self.host = host
        self.port = str(port)
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.alias = alias
        self.collection: Collection | None = None

    def connect(self) -> None:
        if self.uri:
            connections.connect(
                alias=self.alias,
                uri=self.uri,
                token=self.token,
            )
            print(f"Milvus connected: {self.uri}")
        else:
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
            return True
        except MilvusException as exc:
            # Try one-by-one insert to skip only the bad records
            if "exceeds max length" in str(exc).lower() or "length of varchar" in str(exc).lower():
                return self._insert_one_by_one(rows)

            if hasattr(self.collection, "upsert"):
                try:
                    self.collection.upsert(rows)
                    print(f"warning: duplicate primary keys were upserted")
                    return True
                except MilvusException as upsert_exc:
                    if "exceeds max length" in str(upsert_exc).lower() or "length of varchar" in str(upsert_exc).lower():
                        return self._insert_one_by_one(rows)
                    print(f"warning: skip batch after Milvus upsert failed: {upsert_exc}")
                    return False

            print(f"warning: skip batch after Milvus insert failed: {exc}")
            return False

    def _insert_one_by_one(self, rows: List[Dict]) -> bool:
        ok = 0
        for row in rows:
            try:
                self.collection.insert([row])
                ok += 1
            except MilvusException:
                try:
                    self.collection.upsert([row])
                    ok += 1
                except MilvusException:
                    pass
        return ok > 0

    def flush(self) -> None:
        if self.collection is None:
            self.collection = Collection(self.collection_name, using=self.alias)
        self.collection.flush()
        print(f"Milvus collection flushed: {self.collection_name}")

    def create_index(self) -> None:
        if self.collection is None:
            self.collection = Collection(self.collection_name, using=self.alias)
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 2048},
            },
        )
        print(f"IVF_FLAT index created on: {self.collection_name}")

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
