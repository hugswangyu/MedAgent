"""多知识库 RAG engine 缓存管理。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from medlive.rag.engine import RagEngine
from medlive.rag.knowledge_base import KnowledgeBaseMeta, KnowledgeBaseStore
from medlive.rag.metadata_store import MetadataStore
from medlive.rag.settings import Settings


class RagEngineManager:
    """按 kb_id 管理多个独立 LightRAG engine。"""

    def __init__(self, settings: Settings) -> None:
        """绑定基础配置。"""

        self.settings = settings
        self.metadata = MetadataStore(
            Path(settings.user_data_dir).expanduser() / "medlive.db",
            Path(settings.knowledge_bases_dir),
        )
        self.kb_store = KnowledgeBaseStore(self.metadata)
        self._engines: dict[str, RagEngine] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """初始化默认知识库并预热默认 engine。"""

        self.kb_store.initialize()
        await self.get_engine("default")

    async def finalize(self) -> None:
        """关闭所有已缓存 engine。"""

        engines = list(self._engines.values())
        self._engines.clear()
        if engines:
            await asyncio.gather(*(engine.finalize() for engine in engines), return_exceptions=True)

    async def get_engine(self, kb_id: str) -> RagEngine:
        """获取或初始化指定知识库 engine。"""

        if kb_id in self._engines:
            return self._engines[kb_id]
        lock = self._locks.setdefault(kb_id, asyncio.Lock())
        async with lock:
            if kb_id in self._engines:
                return self._engines[kb_id]
            meta = self.kb_store.get(kb_id)
            engine = RagEngine(self._settings_for(meta))
            await engine.initialize()
            self._engines[kb_id] = engine
            return engine

    async def close_engine(self, kb_id: str) -> None:
        """关闭并移除指定知识库 engine。"""

        engine = self._engines.pop(kb_id, None)
        if engine is not None:
            await engine.finalize()

    async def delete_knowledge_base(self, kb_id: str) -> None:
        """关闭 engine 后删除知识库。"""

        await self.close_engine(kb_id)
        self.kb_store.delete(kb_id)

    async def ready_state(self) -> dict[str, Any]:
        """返回服务 ready 状态。"""

        return {
            "initialized": True,
            "provider_configured": self.settings.provider_ready(),
            "llm_model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_dim": self.settings.embedding_dim,
            "user_data_dir": self.settings.absolute_user_data_dir,
            "knowledge_bases_dir": self.settings.knowledge_bases_dir,
            "cached_kb_ids": sorted(self._engines),
        }

    def _settings_for(self, meta: KnowledgeBaseMeta) -> Settings:
        """基于知识库元数据生成独立 workspace 配置。"""

        return replace(
            self.settings,
            kb_id=meta.kb_id,
            kb_name=meta.name,
            working_dir=str(meta.storage_dir),
            upload_dir=str(meta.sources_dir),
            rag_log_dir=str(meta.logs_dir),
            workspace=meta.kb_id,
        )
