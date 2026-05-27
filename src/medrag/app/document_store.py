"""文档管理：内存 job 追踪 + JSON 文档索引。

Jobs 在内存中，重启丢失。文档索引持久化到 ``tmp_data/documents.json``。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .schemas import JobStepItem, DocumentItem

logger = logging.getLogger(__name__)

_DOC_INDEX_FILE = os.path.join("tmp_data", "documents.json")

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


def _load_index() -> list:
    try:
        with open(_DOC_INDEX_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_index(docs: list) -> None:
    folder = os.path.dirname(_DOC_INDEX_FILE)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(_DOC_INDEX_FILE, "w") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)


def get_documents() -> List[DocumentItem]:
    docs = _load_index()
    return [
        DocumentItem(
            filename=d["filename"],
            file_type=d.get("file_type", ""),
            chunk_count=d.get("chunk_count", 0),
        )
        for d in docs
    ]


def add_document(filename: str, file_type: str = "", chunk_count: int = 0) -> None:
    docs = _load_index()
    # 去重
    docs = [d for d in docs if d.get("filename") != filename]
    docs.append({
        "filename": filename,
        "file_type": file_type,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_index(docs)


def remove_document(filename: str) -> bool:
    docs = _load_index()
    filtered = [d for d in docs if d.get("filename") != filename]
    if len(filtered) == len(docs):
        return False
    _save_index(filtered)
    return True


def get_document_by_filename(filename: str) -> Optional[dict]:
    for d in _load_index():
        if d.get("filename") == filename:
            return d
    return None
