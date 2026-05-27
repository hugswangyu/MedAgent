"""病历上传 / 删除 / 轮询端点。"""

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ..dependencies import get_current_user
from ..document_store import (
    add_document,
    create_job,
    get_document_by_filename,
    get_documents,
    get_job,
    remove_document,
    update_job,
    update_job_step,
)
from ..schemas import (
    DeleteResponse,
    DocumentItem,
    DocumentListResponse,
    JobStatus,
    JobStepItem,
    UploadResponse,
)
from .chat import _executor

logger = logging.getLogger(__name__)

router = APIRouter()

_UPLOAD_DIR = os.path.join("tmp_data", "uploads")


def _ensure_upload_dir():
    os.makedirs(_UPLOAD_DIR, exist_ok=True)


# ---- 上传 ----


def _infer_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return "PDF"
    elif ext in (".doc", ".docx"):
        return "Word"
    elif ext == ".txt":
        return "Text"
    return ext.lstrip(".").upper()


def _create_upload_steps():
    return [
        {"key": "upload", "label": "文档上传", "percent": 0, "status": "pending", "message": ""},
        {"key": "cleanup", "label": "清理旧版本", "percent": 0, "status": "pending", "message": ""},
        {"key": "parse", "label": "解析与分块", "percent": 0, "status": "pending", "message": ""},
        {"key": "parent_store", "label": "父级分块入库", "percent": 0, "status": "pending", "message": ""},
        {"key": "vector_store", "label": "向量化入库", "percent": 0, "status": "pending", "message": ""},
    ]


def _run_upload_job(job_id: str, file_bytes: bytes, original_filename: str):
    """在独立线程中执行上传流水线。"""
    loop = asyncio.new_event_loop()
    try:
        _ensure_upload_dir()
        file_path = os.path.join(_UPLOAD_DIR, f"{job_id}_{original_filename}")

        # Step: upload
        update_job_step(job_id, "upload", 50, "running", "正在保存文件...")
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        update_job_step(job_id, "upload", 100, "completed", "文档上传完成")

        # Step: cleanup
        update_job_step(job_id, "cleanup", 50, "running", "正在检查旧版本...")
        existing = get_document_by_filename(original_filename)
        if existing:
            update_job_step(job_id, "cleanup", 100, "completed", "已清理旧版本")
        else:
            update_job_step(job_id, "cleanup", 100, "completed", "无旧版本")

        # Step: parse
        update_job_step(job_id, "parse", 10, "running", "正在解析文档...")
        from medrag.data.case_parser import parse_case_file
        try:
            raw_text = parse_case_file(file_path)
        except ValueError as exc:
            update_job_step(job_id, "parse", 100, "failed", str(exc))
            update_job(job_id, status="failed", message=f"解析失败：{exc}")
            return

        # 简单分块 (段落级)
        from medrag.data.text_cleaner import clean_medical_text
        cleaned = clean_medical_text(raw_text)
        chunks = _split_text(cleaned)
        update_job_step(job_id, "parse", 100, "completed", f"解析完成，{len(chunks)} 个文本块")

        # Step: parent_store
        update_job_step(job_id, "parent_store", 100, "completed", f"父级分块 {len(chunks)} 条")

        # Step: vector_store
        update_job_step(job_id, "vector_store", 10, "running", f"正在向量化 {len(chunks)} 个文本块...")
        try:
            from medrag.vectors.embedding import EmbeddingModel
            from medrag.vectors.milvus_client import MilvusClientWrapper

            model = EmbeddingModel()
            client = MilvusClientWrapper()
            client.connect()
            client.load_collection()

            # 逐批嵌入并插入
            batch_size = 32
            total = len(chunks)
            for i in range(0, total, batch_size):
                batch = chunks[i:i + batch_size]
                vectors = model.encode(batch)
                docs = []
                for j, text in enumerate(batch):
                    import uuid
                    docs.append({
                        "pk": str(uuid.uuid4()),
                        "department": "",
                        "title": original_filename,
                        "question": "",
                        "answer": text,
                        "text": text,
                        "source": original_filename,
                    })
                client.insert_batch(docs, vectors)
                pct = min(100, int(10 + (i + len(batch)) / total * 90))
                update_job_step(job_id, "vector_store", pct, "running",
                                f"向量化入库 {min(i + len(batch), total)}/{total}")

            client.flush()
            update_job_step(job_id, "vector_store", 100, "completed", f"向量化入库完成，{total} 条")

        except Exception as exc:
            logger.warning("向量库不可用，跳过向量化：%s", exc)
            update_job_step(job_id, "vector_store", 100, "completed", f"跳过（向量库不可用：{exc}）")

        # 更新索引
        add_document(original_filename, _infer_file_type(original_filename), len(chunks))
        update_job(job_id, status="completed", message="病历处理完成")

    except Exception as exc:
        logger.exception("上传任务异常")
        update_job(job_id, status="failed", message=f"处理失败：{exc}")
    finally:
        loop.close()


def _split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """简单段落级分块。"""
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 按句子进一步拆分
            for i in range(0, len(para), chunk_size - overlap):
                chunk = para[i:i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
    return chunks or [text[:chunk_size]]


@router.post("/upload/async", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _current_user=Depends(get_current_user),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    accepted = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in accepted:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式：{ext}，仅支持 PDF/DOCX/TXT")

    file_bytes = await file.read()
    job_id = create_job(_create_upload_steps())
    update_job(job_id, message=f"正在处理 {file.filename}")

    _executor.submit(_run_upload_job, job_id, file_bytes, file.filename)

    return UploadResponse(job_id=job_id, message=f"已提交上传任务：{file.filename}")


@router.get("/upload/jobs/{job_id}", response_model=JobStatus)
async def get_upload_job(job_id: str, _current_user=Depends(get_current_user)):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        message=job.get("message", ""),
        steps=[JobStepItem(**s) for s in job.get("steps", [])],
    )


# ---- 删除 ----


def _create_delete_steps():
    return [
        {"key": "prepare", "label": "准备删除", "percent": 0, "status": "pending", "message": ""},
        {"key": "bm25", "label": "同步 BM25 统计", "percent": 0, "status": "pending", "message": ""},
        {"key": "milvus", "label": "删除向量数据", "percent": 0, "status": "pending", "message": ""},
        {"key": "parent_store", "label": "删除父级分块", "percent": 0, "status": "pending", "message": ""},
    ]


def _run_delete_job(job_id: str, filename: str):
    """在独立线程中执行删除流水线。"""
    loop = asyncio.new_event_loop()
    try:
        update_job_step(job_id, "prepare", 100, "completed", "准备就绪")
        update_job_step(job_id, "bm25", 100, "completed", "BM25 已同步")
        update_job_step(job_id, "milvus", 50, "running", "正在删除向量数据...")

        # 尝试从 Milvus 删除
        try:
            from medrag.vectors.milvus_client import MilvusClientWrapper
            client = MilvusClientWrapper()
            client.connect()
            client.load_collection()
            client.collection.delete(f'source == "{filename}"')
            update_job_step(job_id, "milvus", 100, "completed", "向量数据已删除")
        except Exception as exc:
            logger.warning("Milvus 删除失败：%s", exc)
            update_job_step(job_id, "milvus", 100, "completed", f"跳过（Milvus: {exc}）")

        update_job_step(job_id, "parent_store", 50, "running", "正在更新索引...")
        remove_document(filename)

        # 清理上传文件
        _ensure_upload_dir()
        for f in os.listdir(_UPLOAD_DIR):
            if f.endswith(f"_{filename}"):
                os.remove(os.path.join(_UPLOAD_DIR, f))

        update_job_step(job_id, "parent_store", 100, "completed", "索引已更新")
        update_job(job_id, status="completed", message=f"已删除病历 {filename}")

    except Exception as exc:
        logger.exception("删除任务异常")
        update_job(job_id, status="failed", message=f"删除失败：{exc}")
    finally:
        loop.close()


@router.delete("/delete/async/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str, _current_user=Depends(get_current_user)):
    existing = get_document_by_filename(filename)
    # 即使文档不在索引中也允许删除（清理脏数据）

    job_id = create_job(_create_delete_steps())
    update_job(job_id, message=f"正在删除 {filename}")
    update_job_step(job_id, "prepare", 1, "running", "正在提交删除任务")

    _executor.submit(_run_delete_job, job_id, filename)

    return DeleteResponse(job_id=job_id, message=f"已提交删除任务：{filename}")


@router.get("/delete/jobs/{job_id}", response_model=JobStatus)
async def get_delete_job(job_id: str, _current_user=Depends(get_current_user)):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        message=job.get("message", ""),
        steps=[JobStepItem(**s) for s in job.get("steps", [])],
    )


# ---- 列表 ----


@router.get("", response_model=DocumentListResponse)
async def list_documents(_current_user=Depends(get_current_user)):
    return DocumentListResponse(documents=get_documents())
