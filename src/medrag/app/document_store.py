"""文档管理：内存 job 追踪 + JSON 文档索引。

Jobs 在内存中，重启丢失。文档索引持久化到 JSON 文件。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from medrag.config.settings import settings
from medrag.infrastructure.storage import JsonStore

from .schemas import JobStepItem, DocumentItem

logger = logging.getLogger(__name__)

_doc_store = JsonStore(str(settings.documents_index_path))

# ---- 内存 Job 追踪 ----
_jobs: Dict[str, dict] = {}


def create_job(steps: List[dict]) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "message": "",
        "steps": steps,
        "created_at": datetime.now(timezone.utc).timestamp(),
    }
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    job = _jobs.get(job_id)
    if job:
        job.update(kwargs)


def update_job_step(job_id: str, step_key: str, percent: int, status: str = "running", message: str = "") -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    for step in job["steps"]:
        if step["key"] == step_key:
            step["percent"] = max(0, min(100, percent))
            step["status"] = status
            step["message"] = message
            break


# ---- JSON 文档索引 ----


def _read_docs() -> list[dict]:
    docs = _doc_store.read()
    return docs if isinstance(docs, list) else []


def get_documents(username: str | None = None) -> List[DocumentItem]:
    docs = _read_docs()
    if username is not None:
        docs = [d for d in docs if d.get("username") == username]
    return [
        DocumentItem(
            filename=d["filename"],
            file_type=d.get("file_type", ""),
            chunk_count=d.get("chunk_count", 0),
            username=d.get("username", ""),
            document_id=d.get("document_id", ""),
            summary=d.get("summary", ""),
            status=d.get("status", "ready"),
            uploaded_at=d.get("uploaded_at", ""),
        )
        for d in docs
    ]


def add_document(
    filename: str,
    file_type: str = "",
    chunk_count: int = 0,
    username: str = "",
    document_id: str = "",
    summary: str = "",
    status: str = "ready",
) -> None:
    document = {
        "filename": filename,
        "file_type": file_type,
        "chunk_count": chunk_count,
        "username": username,
        "document_id": document_id,
        "summary": summary,
        "status": status,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    def add(docs: dict | list) -> None:
        if not isinstance(docs, list):
            raise TypeError("document index must contain a JSON list")
        docs[:] = [
            d for d in docs
            if not (d.get("filename") == filename and d.get("username", "") == username)
        ]
        docs.append(document)

    _doc_store.update(add, default_factory=list)


def remove_document(filename: str, username: str | None = None) -> bool:
    def remove(docs: dict | list) -> bool:
        if not isinstance(docs, list):
            raise TypeError("document index must contain a JSON list")
        original_count = len(docs)
        docs[:] = [
            d for d in docs
            if not (
                d.get("filename") == filename
                and (username is None or d.get("username") == username)
            )
        ]
        return len(docs) != original_count

    return _doc_store.update(remove, default_factory=list)


def get_document_by_filename(filename: str, username: str | None = None) -> Optional[dict]:
    for d in _read_docs():
        if d.get("filename") == filename and (username is None or d.get("username") == username):
            return d
    return None
