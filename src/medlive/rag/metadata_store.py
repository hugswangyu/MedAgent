"""SQLite 元数据存储。"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_KB_ID = "default"
DEFAULT_KB_NAME = "默认知识库"
_KB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass(frozen=True)
class KnowledgeBaseMeta:
    """单个知识库的元数据和物理目录。"""

    kb_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    owner_user_id: str
    root_dir: Path

    @property
    def storage_dir(self) -> Path:
        """返回该知识库独立 LightRAG storage 目录。"""

        return self.root_dir / "storage"

    @property
    def sources_dir(self) -> Path:
        """返回该知识库原文件目录。"""

        return self.root_dir / "sources"

    @property
    def logs_dir(self) -> Path:
        """返回该知识库日志目录。"""

        return self.root_dir / "logs"


class MetadataStore:
    """使用 SQLite 管理 LiveRAG 产品元数据。"""

    def __init__(self, db_path: Path, knowledge_bases_dir: Path) -> None:
        """绑定数据库文件和知识库根目录。"""

        self.db_path = db_path.expanduser()
        self.knowledge_bases_dir = knowledge_bases_dir.expanduser()

    def initialize(self) -> None:
        """初始化数据库表、目录和默认知识库。"""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_bases_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    kb_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    owner_user_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    source_file_path TEXT NOT NULL,
                    source_file_size INTEGER NOT NULL DEFAULT 0,
                    source_sha256 TEXT NOT NULL DEFAULT '',
                    content_type TEXT NOT NULL DEFAULT '',
                    extension TEXT NOT NULL DEFAULT '',
                    parse_status TEXT NOT NULL DEFAULT 'pending',
                    index_status TEXT NOT NULL DEFAULT 'pending',
                    error_msg TEXT,
                    content_length INTEGER NOT NULL DEFAULT 0,
                    chunks_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(kb_id) REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_documents_kb_updated
                    ON documents(kb_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    parsed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_msg TEXT,
                    FOREIGN KEY(kb_id) REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingest_job_documents (
                    job_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_msg TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, document_id),
                    FOREIGN KEY(job_id) REFERENCES ingest_jobs(job_id) ON DELETE CASCADE,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS session_config (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(knowledge_bases)").fetchall()
            }
            if "owner_user_id" not in columns:
                conn.execute(
                    "ALTER TABLE knowledge_bases ADD COLUMN owner_user_id TEXT NOT NULL DEFAULT ''"
                )
            legacy_owner = os.getenv("LIVERAG_LEGACY_OWNER_USER_ID", "").strip()
            if legacy_owner:
                conn.execute(
                    "UPDATE knowledge_bases SET owner_user_id = ? WHERE owner_user_id = ''",
                    (legacy_owner,),
                )
        self.ensure_default_knowledge_base()

    def ensure_default_knowledge_base(self) -> KnowledgeBaseMeta:
        """确保默认知识库存在。"""

        try:
            return self.get_knowledge_base(DEFAULT_KB_ID)
        except KeyError:
            return self.create_knowledge_base(
                name=DEFAULT_KB_NAME,
                description="",
                kb_id=DEFAULT_KB_ID,
                owner_user_id=os.getenv("LIVERAG_LEGACY_OWNER_USER_ID", "").strip(),
            )

    def list_knowledge_bases(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        """读取全部知识库摘要。"""

        with self._connect() as conn:
            query = """
                SELECT kb.*, COUNT(d.document_id) AS document_count,
                       COALESCE(SUM(d.chunks_count), 0) AS chunk_count
                FROM knowledge_bases kb
                LEFT JOIN documents d ON d.kb_id = kb.kb_id
            """
            params: list[Any] = []
            if owner_user_id is not None:
                query += " WHERE kb.owner_user_id = ?"
                params.append(owner_user_id)
            query += """
                GROUP BY kb.kb_id
                ORDER BY CASE WHEN kb.kb_id = ? THEN 0 ELSE 1 END, kb.created_at
            """
            params.append(DEFAULT_KB_ID)
            rows = conn.execute(query, params).fetchall()
        return [self._kb_public_from_row(row) for row in rows]

    def create_knowledge_base(
        self,
        *,
        name: str,
        description: str = "",
        kb_id: str | None = None,
        owner_user_id: str = "",
    ) -> KnowledgeBaseMeta:
        """创建知识库元数据和目录。"""

        clean_name = name.strip()
        if not clean_name:
            raise ValueError("knowledge base name cannot be empty")
        new_id = kb_id or f"kb_{self._random_id()}"
        self.validate_kb_id(new_id)
        now = self._now_iso()
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO knowledge_bases(
                        kb_id, name, description, owner_user_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (new_id, clean_name, description.strip(), owner_user_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"knowledge base already exists: {new_id}") from exc
        meta = self.get_knowledge_base(new_id)
        self.ensure_knowledge_base_dirs(meta)
        return meta

    def get_knowledge_base(
        self, kb_id: str, owner_user_id: str | None = None
    ) -> KnowledgeBaseMeta:
        """读取单个知识库。"""

        self.validate_kb_id(kb_id)
        with self._connect() as conn:
            query = "SELECT * FROM knowledge_bases WHERE kb_id = ?"
            params: list[Any] = [kb_id]
            if owner_user_id is not None:
                query += " AND owner_user_id = ?"
                params.append(owner_user_id)
            row = conn.execute(query, params).fetchone()
        if row is None:
            raise KeyError(f"knowledge base not found: {kb_id}")
        meta = self._kb_meta_from_row(row)
        self.ensure_knowledge_base_dirs(meta)
        return meta

    def update_knowledge_base(
        self,
        kb_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBaseMeta:
        """更新知识库名称和描述。"""

        current = self.get_knowledge_base(kb_id)
        new_name = current.name if name is None else name.strip()
        if not new_name:
            raise ValueError("knowledge base name cannot be empty")
        new_description = current.description if description is None else description.strip()
        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE knowledge_bases
                SET name = ?, description = ?, updated_at = ?
                WHERE kb_id = ?
                """,
                (new_name, new_description, now, kb_id),
            )
        return self.get_knowledge_base(kb_id)

    def delete_knowledge_base_metadata(self, kb_id: str) -> None:
        """删除知识库相关 SQLite 元数据。"""

        if kb_id == DEFAULT_KB_ID:
            raise ValueError("default knowledge base cannot be deleted")
        self.get_knowledge_base(kb_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,))

    def public_knowledge_base_detail(
        self, kb_id: str, owner_user_id: str | None = None
    ) -> dict[str, Any]:
        """返回单个知识库公开详情。"""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT kb.*, COUNT(d.document_id) AS document_count,
                       COALESCE(SUM(d.chunks_count), 0) AS chunk_count
                FROM knowledge_bases kb
                LEFT JOIN documents d ON d.kb_id = kb.kb_id
                WHERE kb.kb_id = ?
                  AND (? IS NULL OR kb.owner_user_id = ?)
                GROUP BY kb.kb_id
                """,
                (kb_id, owner_user_id, owner_user_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"knowledge base not found: {kb_id}")
        return self._kb_public_from_row(row)

    def create_document(
        self,
        *,
        document_id: str,
        kb_id: str,
        original_filename: str,
        source_file_path: Path,
        source_file_size: int,
        source_sha256: str,
        content_type: str,
        extension: str,
    ) -> dict[str, Any]:
        """创建文档元数据记录。"""

        self.get_knowledge_base(kb_id)
        now = self._now_iso()
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO documents(
                        document_id, kb_id, original_filename, source_file_path,
                        source_file_size, source_sha256, content_type, extension,
                        parse_status, index_status, error_msg, content_length,
                        chunks_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'pending', NULL, 0, 0, ?, ?)
                    """,
                    (
                        document_id,
                        kb_id,
                        original_filename,
                        str(source_file_path),
                        source_file_size,
                        source_sha256,
                        content_type,
                        extension,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"document already exists: {document_id}") from exc
        return self.get_document(kb_id, document_id)

    def mark_document_parsed(self, kb_id: str, document_id: str, *, content_length: int) -> None:
        """标记文档解析成功。"""

        self._update_document(
            kb_id,
            document_id,
            parse_status="parsed",
            error_msg=None,
            content_length=content_length,
        )

    def mark_document_failed(self, kb_id: str, document_id: str, *, error_msg: str) -> None:
        """标记文档解析或索引失败。"""

        self._update_document(
            kb_id,
            document_id,
            parse_status="failed",
            index_status="failed",
            error_msg=error_msg,
        )

    def mark_document_indexing(self, kb_id: str, document_id: str) -> None:
        """标记文档已进入索引队列。"""

        self._update_document(kb_id, document_id, index_status="processing")

    def update_document_index_status(
        self,
        kb_id: str,
        document_id: str,
        *,
        index_status: str,
        chunks_count: int | None = None,
        error_msg: str | None = None,
    ) -> None:
        """同步 LightRAG 文档索引状态。"""

        values: dict[str, Any] = {"index_status": index_status, "error_msg": error_msg}
        if chunks_count is not None:
            values["chunks_count"] = chunks_count
        self._update_document(kb_id, document_id, **values)

    def list_documents(self, kb_id: str, *, page: int, page_size: int) -> dict[str, Any]:
        """分页读取知识库文档。"""

        meta = self.get_knowledge_base(kb_id)
        offset = (page - 1) * page_size
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS count FROM documents WHERE kb_id = ?",
                (kb_id,),
            ).fetchone()["count"]
            rows = conn.execute(
                """
                SELECT d.*, kb.name AS kb_name
                FROM documents d
                JOIN knowledge_bases kb ON kb.kb_id = d.kb_id
                WHERE d.kb_id = ?
                ORDER BY d.updated_at DESC, d.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (kb_id, page_size, offset),
            ).fetchall()
        total_pages = (total + page_size - 1) // page_size if total else 0
        return {
            "documents": [self._document_public_from_row(row) for row in rows],
            "kb_id": meta.kb_id,
            "kb_name": meta.name,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1 and total_pages > 0,
        }

    def get_document(self, kb_id: str, document_id: str) -> dict[str, Any]:
        """读取单个文档元数据。"""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.*, kb.name AS kb_name
                FROM documents d
                JOIN knowledge_bases kb ON kb.kb_id = d.kb_id
                WHERE d.kb_id = ? AND d.document_id = ?
                """,
                (kb_id, document_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"document not found: {document_id}")
        return self._document_public_from_row(row)

    def delete_document_metadata(self, kb_id: str, document_id: str) -> dict[str, Any]:
        """删除单个文档元数据。"""

        document = self.get_document(kb_id, document_id)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM documents WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            )
        return document

    def clear_documents_metadata(self, kb_id: str) -> None:
        """清空知识库下全部文档和任务元数据。"""

        self.get_knowledge_base(kb_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE kb_id = ?", (kb_id,))
            conn.execute("DELETE FROM ingest_jobs WHERE kb_id = ?", (kb_id,))

    def create_job(self, *, job_id: str, kb_id: str, total_files: int) -> None:
        """创建导入任务。"""

        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingest_jobs(
                    job_id, kb_id, status, total_files, parsed_count,
                    failed_count, created_at, updated_at, error_msg
                ) VALUES (?, ?, 'processing', ?, 0, 0, ?, ?, NULL)
                """,
                (job_id, kb_id, total_files, now, now),
            )

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        parsed_count: int | None = None,
        failed_count: int | None = None,
        error_msg: str | None = None,
    ) -> None:
        """更新导入任务状态。"""

        current = self.get_job(job_id)
        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, parsed_count = ?, failed_count = ?, error_msg = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    current["parsed_count"] if parsed_count is None else parsed_count,
                    current["failed_count"] if failed_count is None else failed_count,
                    error_msg,
                    now,
                    job_id,
                ),
            )

    def link_job_document(
        self,
        *,
        job_id: str,
        document_id: str,
        status: str,
        error_msg: str | None = None,
    ) -> None:
        """关联导入任务和文档。"""

        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingest_job_documents(
                    job_id, document_id, status, error_msg, created_at, updated_at
                ) VALUES (?, ?, ?, ?, COALESCE((
                    SELECT created_at FROM ingest_job_documents WHERE job_id = ? AND document_id = ?
                ), ?), ?)
                """,
                (job_id, document_id, status, error_msg, job_id, document_id, now, now),
            )

    def get_job(self, job_id: str) -> dict[str, Any]:
        """读取导入任务。"""

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        return dict(row)

    def job_detail(self, kb_id: str, job_id: str) -> dict[str, Any]:
        """读取导入任务和关联文档。"""

        with self._connect() as conn:
            job = conn.execute(
                "SELECT * FROM ingest_jobs WHERE kb_id = ? AND job_id = ?",
                (kb_id, job_id),
            ).fetchone()
            if job is None:
                raise KeyError(f"job not found: {job_id}")
            rows = conn.execute(
                """
                SELECT jd.status AS job_document_status, jd.error_msg AS job_error_msg,
                       d.*, kb.name AS kb_name
                FROM ingest_job_documents jd
                JOIN documents d ON d.document_id = jd.document_id
                JOIN knowledge_bases kb ON kb.kb_id = d.kb_id
                WHERE jd.job_id = ?
                ORDER BY d.created_at
                """,
                (job_id,),
            ).fetchall()
        payload = dict(job)
        payload["documents"] = [self._document_public_from_row(row) | {
            "job_document_status": row["job_document_status"],
            "job_error_msg": row["job_error_msg"],
        } for row in rows]
        payload["total"] = len(rows)
        return payload

    def get_session_config(self, key: str) -> dict[str, Any]:
        """读取 JSON 会话配置。"""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM session_config WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return {}
        try:
            data = json.loads(row["value_json"])
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def set_session_config(self, key: str, value: dict[str, Any]) -> None:
        """写入 JSON 会话配置。"""

        now = self._now_iso()
        raw = json.dumps(value, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_config(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,
                                                  updated_at = excluded.updated_at
                """,
                (key, raw, now),
            )

    def source_document_dir(self, kb_id: str, document_id: str) -> Path:
        """返回文档原文件目录。"""

        meta = self.get_knowledge_base(kb_id)
        return meta.sources_dir / document_id

    def _update_document(self, kb_id: str, document_id: str, **values: Any) -> None:
        """按字段更新文档记录。"""

        if not values:
            return
        allowed = {
            "parse_status",
            "index_status",
            "error_msg",
            "content_length",
            "chunks_count",
        }
        fields = {key: value for key, value in values.items() if key in allowed}
        if not fields:
            return
        fields["updated_at"] = self._now_iso()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        params = [*fields.values(), kb_id, document_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE documents SET {assignments} WHERE kb_id = ? AND document_id = ?",
                params,
            )

    def _connect(self) -> sqlite3.Connection:
        """创建 SQLite 连接。"""

        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _kb_meta_from_row(self, row: sqlite3.Row) -> KnowledgeBaseMeta:
        """把 DB 行转换成知识库元数据。"""

        kb_id = str(row["kb_id"])
        return KnowledgeBaseMeta(
            kb_id=kb_id,
            name=str(row["name"]),
            description=str(row["description"] or ""),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            owner_user_id=str(row["owner_user_id"] or ""),
            root_dir=self.knowledge_bases_dir / kb_id,
        )

    def _kb_public_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        """把 DB 行转换成公开知识库字段。"""

        return {
            "kb_id": row["kb_id"],
            "name": row["name"],
            "description": row["description"] or "",
            "owner_user_id": row["owner_user_id"] or "",
            "document_count": int(row["document_count"] or 0),
            "chunk_count": int(row["chunk_count"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _document_public_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        """把 DB 行转换成公开文档字段。"""

        source_path = Path(str(row["source_file_path"] or ""))
        parse_status = str(row["parse_status"])
        index_status = str(row["index_status"])
        status = index_status
        if parse_status == "failed":
            status = "parse_failed"
        elif index_status == "failed":
            status = "failed"
        return {
            "document_id": row["document_id"],
            "kb_id": row["kb_id"],
            "kb_name": row["kb_name"],
            "original_filename": row["original_filename"],
            "file_path": row["original_filename"],
            "source_file_path": str(source_path),
            "source_file_exists": source_path.is_file(),
            "source_file_size": int(row["source_file_size"] or 0),
            "source_sha256": row["source_sha256"] or "",
            "content_type": row["content_type"] or "",
            "extension": row["extension"] or "",
            "parse_status": parse_status,
            "index_status": index_status,
            "status": status,
            "error_msg": row["error_msg"],
            "content_summary": "",
            "content_length": int(row["content_length"] or 0),
            "chunks_count": int(row["chunks_count"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def ensure_knowledge_base_dirs(meta: KnowledgeBaseMeta) -> None:
        """创建知识库运行目录。"""

        for directory in (meta.root_dir, meta.storage_dir, meta.sources_dir, meta.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_kb_id(kb_id: str) -> None:
        """校验知识库 ID。"""

        if not kb_id or not _KB_ID_RE.match(kb_id):
            raise ValueError("invalid knowledge base id")

    @staticmethod
    def _now_iso() -> str:
        """返回 UTC ISO 时间。"""

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _random_id() -> str:
        """生成短随机 ID。"""

        import uuid

        return uuid.uuid4().hex[:12]
