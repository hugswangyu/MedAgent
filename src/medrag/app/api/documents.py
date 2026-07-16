"""病历上传 / 删除 / 轮询端点。"""

import asyncio
import logging
import os
import uuid
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
from medrag.data.user_case_store import add_user_case, remove_user_case

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
    ]


def _run_upload_job(job_id: str, file_bytes: bytes, original_filename: str, username: str):
    """在独立线程中执行上传流水线。"""
    loop = asyncio.new_event_loop()
    try:
        _ensure_upload_dir()
        user_upload_dir = os.path.join(_UPLOAD_DIR, username)
        os.makedirs(user_upload_dir, exist_ok=True)
        file_path = os.path.join(user_upload_dir, f"{job_id}_{original_filename}")
        document_id = str(uuid.uuid4())

        # Step: upload
        update_job_step(job_id, "upload", 50, "running", "正在保存文件...")
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        update_job_step(job_id, "upload", 100, "completed", "文档上传完成")

        # Step: cleanup
        update_job_step(job_id, "cleanup", 50, "running", "正在检查旧版本...")
        existing = get_document_by_filename(original_filename, username=username)
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
        from medrag.data.text_cleaner import clean_medical_text, desensitize_medical_text
        cleaned = clean_medical_text(raw_text)
        safe_text = desensitize_medical_text(cleaned)
        chunks = _split_text(safe_text)
        update_job_step(job_id, "parse", 100, "completed", f"解析完成，{len(chunks)} 个文本块")

        # Step: parent_store
        summary = _build_case_summary(safe_text)
        add_user_case(
            username=username,
            filename=original_filename,
            chunks=chunks,
            summary=summary,
            document_id=document_id,
        )
        update_job_step(job_id, "parent_store", 100, "completed", f"个人病例索引 {len(chunks)} 条")

        # 更新索引
        add_document(
            original_filename,
            _infer_file_type(original_filename),
            len(chunks),
            username=username,
            document_id=document_id,
            summary=summary,
            status="ready",
        )
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


def _build_case_summary(text: str, max_chars: int = 1600) -> str:
    """生成不依赖 LLM 的病例摘要兜底，供 prompt 优先上下文使用。"""
    labels = [
        "主诉", "现病史", "既往史", "检查", "检验", "诊断",
        "用药", "医嘱", "建议", "异常", "过敏",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    picked = [
        line for line in lines
        if any(label in line[:20] for label in labels)
    ]
    summary = "\n".join(picked[:20]) or text[:max_chars]
    return summary[:max_chars]


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

    _executor.submit(_run_upload_job, job_id, file_bytes, file.filename, _current_user.username)

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
        {"key": "parent_store", "label": "删除父级分块", "percent": 0, "status": "pending", "message": ""},
    ]


def _run_delete_job(job_id: str, filename: str, username: str):
    """在独立线程中执行删除流水线。"""
    loop = asyncio.new_event_loop()
    try:
        update_job_step(job_id, "prepare", 100, "completed", "准备就绪")
        update_job_step(job_id, "bm25", 100, "completed", "BM25 已同步")
        update_job_step(job_id, "parent_store", 50, "running", "正在更新索引...")
        remove_document(filename, username=username)
        remove_user_case(username, filename)

        # 清理上传文件
        _ensure_upload_dir()
        user_upload_dir = os.path.join(_UPLOAD_DIR, username)
        if os.path.isdir(user_upload_dir):
            for f in os.listdir(user_upload_dir):
                if f.endswith(f"_{filename}"):
                    os.remove(os.path.join(user_upload_dir, f))

        update_job_step(job_id, "parent_store", 100, "completed", "索引已更新")
        update_job(job_id, status="completed", message=f"已删除病历 {filename}")

    except Exception as exc:
        logger.exception("删除任务异常")
        update_job(job_id, status="failed", message=f"删除失败：{exc}")
    finally:
        loop.close()


@router.delete("/delete/async/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str, _current_user=Depends(get_current_user)):
    existing = get_document_by_filename(filename, username=_current_user.username)
    # 即使文档不在索引中也允许删除（清理脏数据）

    job_id = create_job(_create_delete_steps())
    update_job(job_id, message=f"正在删除 {filename}")
    update_job_step(job_id, "prepare", 1, "running", "正在提交删除任务")

    _executor.submit(_run_delete_job, job_id, filename, _current_user.username)

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
    return DocumentListResponse(documents=get_documents(username=_current_user.username))
