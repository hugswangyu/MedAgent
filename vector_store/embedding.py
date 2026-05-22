"""Sentence-transformers embedding wrapper."""

from __future__ import annotations

from typing import List

import torch
from sentence_transformers import SentenceTransformer

from config.settings import settings


BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class EmbeddingModel:
    def __init__(self, model_name: str = settings.embedding_model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(
            f"Embedding model loaded: {self.model_name}, "
            f"device={self.device}, dim={self.embedding_dim}"
        )

    def _prepare_texts(self, texts: List[str], is_query: bool) -> List[str]:
        if is_query and "bge" in self.model_name.lower():
            return [BGE_QUERY_PREFIX + text for text in texts]
        return texts

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        is_query: bool = False,
    ) -> List[List[float]]:
        if not texts:
            return []

        prepared_texts = self._prepare_texts(texts, is_query=is_query)
        embeddings = self.model.encode(
            prepared_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_one(self, text: str, is_query: bool = False) -> List[float]:
        embeddings = self.encode([text], is_query=is_query)
        return embeddings[0] if embeddings else []

