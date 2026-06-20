"""Milvus 客户端封装，用于医疗问答向量存储。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

try:
    from pymilvus import MilvusClient, DataType
    from pymilvus.exceptions import MilvusException
except Exception:  # pragma: no cover - optional runtime dependency
    MilvusClient = None  # type: ignore[assignment,misc]
    DataType = None  # type: ignore[assignment]

    class MilvusException(Exception):  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)
from medrag.config.settings import settings


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
        alias: str = "default",  # kept for API compatibility, unused
    ):
        self.host = host
        self.port = str(port)
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self._client: MilvusClient | None = None

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.uri:
            self._client = MilvusClient(uri=self.uri, token=self.token)
            logger.info("Milvus connected: %s", self.uri)
        else:
            self._client = MilvusClient(host=self.host, port=int(self.port))
            logger.info("Milvus connected: %s:%s", self.host, self.port)

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self.connect()
        return self._client

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def create_collection(self, embedding_dim: int, recreate: bool = False) -> "MilvusClientWrapper":
        if recreate and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info("Dropped Milvus collection: %s", self.collection_name)

        if self.client.has_collection(self.collection_name):
            logger.info("Using existing Milvus collection: %s", self.collection_name)
            return self

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk",         DataType.VARCHAR,      is_primary=True, max_length=VARCHAR_LIMITS["pk"])
        schema.add_field("department", DataType.VARCHAR,      max_length=VARCHAR_LIMITS["department"])
        schema.add_field("title",      DataType.VARCHAR,      max_length=VARCHAR_LIMITS["title"])
        schema.add_field("question",   DataType.VARCHAR,      max_length=VARCHAR_LIMITS["question"])
        schema.add_field("answer",     DataType.VARCHAR,      max_length=VARCHAR_LIMITS["answer"])
        schema.add_field("text",       DataType.VARCHAR,      max_length=VARCHAR_LIMITS["text"])
        schema.add_field("source",     DataType.VARCHAR,      max_length=VARCHAR_LIMITS["source"])
        schema.add_field("embedding",  DataType.FLOAT_VECTOR, dim=embedding_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 2048},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created Milvus collection: %s, dim=%d", self.collection_name, embedding_dim)
        return self

    def load_collection(self) -> "MilvusClientWrapper":
        self.client.load_collection(self.collection_name)
        logger.info("Milvus collection loaded: %s", self.collection_name)
        return self

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def insert_batch(self, docs: List[Dict], embeddings: List[List[float]]) -> bool:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must have the same length")
        if not docs:
            return True

        rows = []
        for doc, embedding in zip(docs, embeddings):
            pk = doc.get("id") or doc.get("pk")
            if pk is None:
                raise ValueError("each doc must include either 'id' or 'pk'")
            rows.append(
                {
                    "pk":         self._clip(str(pk), "pk"),
                    "department": self._clip(doc.get("department", ""), "department"),
                    "title":      self._clip(doc.get("title", ""), "title"),
                    "question":   self._clip(doc.get("question", ""), "question"),
                    "answer":     self._clip(doc.get("answer", ""), "answer"),
                    "text":       self._clip(doc.get("text", ""), "text"),
                    "source":     self._clip(doc.get("source", "cmedqa2"), "source"),
                    "embedding":  embedding,
                }
            )

        try:
            self.client.insert(collection_name=self.collection_name, data=rows)
            return True
        except MilvusException as exc:
            if "exceeds max length" in str(exc).lower() or "length of varchar" in str(exc).lower():
                return self._insert_one_by_one(rows)
            try:
                self.client.upsert(collection_name=self.collection_name, data=rows)
                logger.warning("Duplicate primary keys were upserted")
                return True
            except MilvusException as upsert_exc:
                if "exceeds max length" in str(upsert_exc).lower() or "length of varchar" in str(upsert_exc).lower():
                    return self._insert_one_by_one(rows)
                logger.warning("Skip batch after Milvus upsert failed: %s", upsert_exc)
                return False

    def _insert_one_by_one(self, rows: List[Dict]) -> bool:
        ok = 0
        for row in rows:
            try:
                self.client.insert(collection_name=self.collection_name, data=[row])
                ok += 1
            except MilvusException:
                try:
                    self.client.upsert(collection_name=self.collection_name, data=[row])
                    ok += 1
                except MilvusException:
                    pass
        return ok > 0

    def flush(self) -> None:
        self.client.flush(collection_name=self.collection_name)
        logger.info("Milvus collection flushed: %s", self.collection_name)

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def create_index(self) -> None:
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 2048},
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        logger.info("IVF_FLAT index created on: %s", self.collection_name)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """向量检索，返回 hit 列表（每个 hit 含 id/distance/entity）。"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
            limit=top_k,
            filter=expr or "",
            output_fields=output_fields or ["pk", "department", "title", "question", "answer", "text"],
        )
        return results[0] if results else []

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _clip(value: object, field_name: str) -> str:
        text = "" if value is None else str(value)
        return text[: VARCHAR_LIMITS[field_name]]
