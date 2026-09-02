"""知识库元数据和物理隔离目录管理。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from medlive.rag.metadata_store import (
    DEFAULT_KB_ID,
    KnowledgeBaseMeta,
    MetadataStore,
)


class KnowledgeBaseStore:
    """基于 SQLite 管理多个物理隔离知识库。"""

    def __init__(self, metadata: MetadataStore) -> None:
        """绑定 SQLite 元数据存储。"""

        self.metadata = metadata
        self.root_dir = metadata.knowledge_bases_dir

    def initialize(self) -> None:
        """初始化元数据和默认知识库。"""

        self.metadata.initialize()

    def ensure_default(self) -> KnowledgeBaseMeta:
        """确保默认知识库存在。"""

        return self.metadata.ensure_default_knowledge_base()

    def list(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        """返回全部知识库摘要。"""

        return self.metadata.list_knowledge_bases(owner_user_id)

    def create(
        self,
        *,
        name: str,
        description: str = "",
        kb_id: str | None = None,
        owner_user_id: str = "",
    ) -> KnowledgeBaseMeta:
        """创建知识库。"""

        return self.metadata.create_knowledge_base(
            name=name,
            description=description,
            kb_id=kb_id,
            owner_user_id=owner_user_id,
        )

    def get(self, kb_id: str) -> KnowledgeBaseMeta:
        """读取知识库元数据。"""

        return self.metadata.get_knowledge_base(kb_id)

    def update(self, kb_id: str, *, name: str | None = None, description: str | None = None) -> KnowledgeBaseMeta:
        """更新知识库元数据。"""

        return self.metadata.update_knowledge_base(kb_id, name=name, description=description)

    def delete(self, kb_id: str) -> None:
        """删除知识库目录和元数据。"""

        if kb_id == DEFAULT_KB_ID:
            raise ValueError("default knowledge base cannot be deleted")
        meta = self.get(kb_id)
        self.metadata.delete_knowledge_base_metadata(kb_id)
        shutil.rmtree(meta.root_dir, ignore_errors=True)

    def public_detail(self, kb_id: str) -> dict[str, Any]:
        """返回单个知识库详情。"""

        return self.metadata.public_knowledge_base_detail(kb_id)

    def source_document_dir(self, kb_id: str, document_id: str) -> Path:
        """返回文档原文件目录。"""

        return self.metadata.source_document_dir(kb_id, document_id)
