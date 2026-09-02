"""内部 LightRAG Core Service，多知识库物理隔离实现。"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from lightrag.utils import generate_track_id
from pydantic import BaseModel, Field

from medlive.rag.doc_parser import parse_file_content
from medlive.rag.engine_manager import RagEngineManager
from medlive.rag.metadata_store import DEFAULT_KB_ID
from medlive.rag.settings import Settings

from .schemas import Envelope, QueryRequest, TextDocumentRequest

settings = Settings()
manager = RagEngineManager(settings)


class KnowledgeBaseCreateRequest(BaseModel):
    """创建知识库请求。"""

    name: str = Field(min_length=1)
    description: str = ""
    owner_user_id: str = ""


class KnowledgeBasePatchRequest(BaseModel):
    """更新知识库请求。"""

    name: str | None = None
    description: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动和关闭 RAG engine manager。"""

    await manager.initialize()
    try:
        yield
    finally:
        await manager.finalize()


app = FastAPI(
    title="LightRAG Core Service",
    version="0.2.0",
    description="A lightweight multi-knowledge-base service around lightrag-hku core APIs.",
    lifespan=lifespan,
)


def envelope(
    *,
    request_id: str,
    data: dict[str, Any] | list[Any] | None = None,
    metrics: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    """生成统一 envelope。"""

    return Envelope(
        request_id=request_id,
        status=status,  # type: ignore[arg-type]
        data=data,
        metrics=metrics or {},
        error=error,
    ).model_dump()


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """校验内部 RAG 服务密钥。"""

    if not settings.api_key:
        return
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    if x_api_key != settings.api_key and bearer != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    """兜底异常响应。"""

    request_id = str(uuid.uuid4())
    return JSONResponse(
        status_code=500,
        content=envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        ),
    )


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    """健康检查。"""

    return envelope(request_id=str(uuid.uuid4()), data={"service": "ok"})


@app.get("/v1/readyz", dependencies=[Depends(require_api_key)])
async def readyz() -> dict[str, Any]:
    """RAG 服务 ready 检查。"""

    state = await manager.ready_state()
    ready = bool(state["initialized"] and state["provider_configured"])
    return envelope(request_id=str(uuid.uuid4()), data={"ready": ready, **state})


@app.get("/v1/knowledge-bases", dependencies=[Depends(require_api_key)])
async def knowledge_bases() -> dict[str, Any]:
    """列出知识库。"""

    items = manager.kb_store.list()
    return envelope(request_id=str(uuid.uuid4()), data={"knowledge_bases": items, "total": len(items)})


@app.post("/v1/knowledge-bases", dependencies=[Depends(require_api_key)])
async def create_knowledge_base(request: KnowledgeBaseCreateRequest) -> dict[str, Any]:
    """创建知识库。"""

    request_id = str(uuid.uuid4())
    try:
        meta = manager.kb_store.create(
            name=request.name,
            description=request.description,
            owner_user_id=request.owner_user_id,
        )
        await manager.get_engine(meta.kb_id)
        return envelope(request_id=request_id, data=manager.kb_store.public_detail(meta.kb_id))
    except ValueError as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": "KnowledgeBaseValidationError", "message": str(exc)},
        )


@app.get("/v1/knowledge-bases/{kb_id}", dependencies=[Depends(require_api_key)])
async def knowledge_base_detail(kb_id: str) -> dict[str, Any]:
    """读取知识库详情。"""

    request_id = str(uuid.uuid4())
    try:
        return envelope(request_id=request_id, data=manager.kb_store.public_detail(kb_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/v1/knowledge-bases/{kb_id}", dependencies=[Depends(require_api_key)])
async def patch_knowledge_base(kb_id: str, request: KnowledgeBasePatchRequest) -> dict[str, Any]:
    """更新知识库元数据。"""

    request_id = str(uuid.uuid4())
    try:
        meta = manager.kb_store.update(kb_id, name=request.name, description=request.description)
        engine = await manager.get_engine(kb_id)
        engine.settings = manager._settings_for(meta)
        return envelope(request_id=request_id, data=manager.kb_store.public_detail(kb_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": "KnowledgeBaseValidationError", "message": str(exc)},
        )


@app.delete("/v1/knowledge-bases/{kb_id}", dependencies=[Depends(require_api_key)])
async def delete_knowledge_base(kb_id: str) -> dict[str, Any]:
    """删除知识库。"""

    request_id = str(uuid.uuid4())
    if kb_id == DEFAULT_KB_ID:
        raise HTTPException(status_code=409, detail="default knowledge base cannot be deleted")
    try:
        await manager.delete_knowledge_base(kb_id)
        return envelope(request_id=request_id, data={"deleted": True, "kb_id": kb_id})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/knowledge-bases/{kb_id}/ready", dependencies=[Depends(require_api_key)])
async def knowledge_base_ready(kb_id: str) -> dict[str, Any]:
    """预热并返回指定知识库 ready 状态。"""

    request_id = str(uuid.uuid4())
    try:
        engine = await manager.get_engine(kb_id)
        return envelope(request_id=request_id, data=engine.ready_state())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/knowledge-bases/{kb_id}/query/defaults", dependencies=[Depends(require_api_key)])
async def query_defaults(kb_id: str) -> dict[str, Any]:
    """返回指定知识库查询默认配置。"""

    engine = await manager.get_engine(kb_id)
    return envelope(request_id=str(uuid.uuid4()), data=engine.defaults())


@app.get("/v1/knowledge-bases/{kb_id}/overview", dependencies=[Depends(require_api_key)])
async def knowledge_overview(
    kb_id: str,
    entity_limit: int = Query(default=20, ge=1, le=100),
    relation_limit: int = Query(default=12, ge=1, le=100),
    document_limit: int = Query(default=10, ge=1, le=100),
    topic_limit: int = Query(default=8, ge=1, le=100),
) -> dict[str, Any]:
    """返回指定知识库概览。"""

    request_id = str(uuid.uuid4())
    try:
        engine = await manager.get_engine(kb_id)
        data = await engine.knowledge_overview(
            entity_limit=entity_limit,
            relation_limit=relation_limit,
            document_limit=document_limit,
            topic_limit=topic_limit,
        )
        return envelope(request_id=request_id, data=data)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.post("/v1/knowledge-bases/{kb_id}/query/context", dependencies=[Depends(require_api_key)])
async def query_context(kb_id: str, request: QueryRequest) -> dict[str, Any]:
    """只查询指定知识库上下文。"""

    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        engine = await manager.get_engine(kb_id)
        data, metrics = await engine.query_context(
            request.query,
            request.profile,
            request.merged_options(),
            request.merged_conversation(),
        )
        metrics["request_total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return envelope(request_id=request_id, data=data, metrics=metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.post("/v1/knowledge-bases/{kb_id}/query/data", dependencies=[Depends(require_api_key)])
async def query_data(kb_id: str, request: QueryRequest) -> dict[str, Any]:
    """只查询指定知识库结构化数据。"""

    request_id = str(uuid.uuid4())
    try:
        engine = await manager.get_engine(kb_id)
        data, metrics = await engine.query_data(
            request.query,
            request.profile,
            request.merged_options(),
            request.merged_conversation(),
        )
        return envelope(request_id=request_id, data=data, metrics=metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.post("/v1/knowledge-bases/{kb_id}/query/answer", dependencies=[Depends(require_api_key)])
async def query_answer(kb_id: str, request: QueryRequest) -> dict[str, Any]:
    """只查询指定知识库并生成答案。"""

    request_id = str(uuid.uuid4())
    try:
        engine = await manager.get_engine(kb_id)
        data, metrics = await engine.query_answer(
            request.query,
            request.profile,
            request.merged_options(),
            request.merged_conversation(),
        )
        return envelope(request_id=request_id, data=data, metrics=metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.post("/v1/knowledge-bases/{kb_id}/documents/text", dependencies=[Depends(require_api_key)])
async def documents_text(kb_id: str, request: TextDocumentRequest) -> dict[str, Any]:
    """向指定知识库导入文本，并保存原始文本文件。"""

    request_id = str(uuid.uuid4())
    try:
        meta = manager.kb_store.get(kb_id)
        document_id = _clean_document_id(request.document_id) if request.document_id else _new_document_id()
        _ensure_document_not_exists(kb_id, document_id)
        filename = _safe_filename(request.file_source or f"{document_id}.txt")
        raw = request.text.encode("utf-8")
        source_path = _write_source_file(kb_id, document_id, filename, raw)
        manager.metadata.create_document(
            document_id=document_id,
            kb_id=kb_id,
            original_filename=filename,
            source_file_path=source_path,
            source_file_size=len(raw),
            source_sha256=_sha256(raw),
            content_type="text/plain; charset=utf-8",
            extension=Path(filename).suffix.lower() or ".txt",
        )
        manager.metadata.mark_document_parsed(kb_id, document_id, content_length=len(request.text))

        track_id = generate_track_id("insert")
        manager.metadata.create_job(job_id=track_id, kb_id=kb_id, total_files=1)
        manager.metadata.link_job_document(job_id=track_id, document_id=document_id, status="processing")
        engine = await manager.get_engine(kb_id)
        try:
            await engine.enqueue_documents(
                texts=[request.text],
                file_sources=[filename],
                document_ids=[document_id],
                track_id=track_id,
            )
            manager.metadata.mark_document_indexing(kb_id, document_id)
            manager.metadata.update_job(track_id, status="processing", parsed_count=1, failed_count=0)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            manager.metadata.update_document_index_status(
                kb_id,
                document_id,
                index_status="failed",
                error_msg=error_msg,
            )
            manager.metadata.link_job_document(
                job_id=track_id,
                document_id=document_id,
                status="failed",
                error_msg=error_msg,
            )
            manager.metadata.update_job(track_id, status="failed", parsed_count=1, failed_count=1, error_msg=error_msg)
            return envelope(
                request_id=request_id,
                status="error",
                data={"document": manager.metadata.get_document(kb_id, document_id), "track_id": track_id},
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        return envelope(
            request_id=request_id,
            data={
                "track_id": track_id,
                "processing_mode": "async",
                "kb_id": meta.kb_id,
                "kb_name": meta.name,
                "document": manager.metadata.get_document(kb_id, document_id),
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": "DocumentValidationError", "message": str(exc)},
        )
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.post("/v1/knowledge-bases/{kb_id}/documents/files", dependencies=[Depends(require_api_key)])
async def documents_files(
    kb_id: str,
    files: Annotated[list[UploadFile], File(...)],
) -> dict[str, Any]:
    """向指定知识库上传原文件、解析并导入。"""

    request_id = str(uuid.uuid4())
    try:
        meta = manager.kb_store.get(kb_id)
        track_id = generate_track_id("insert")
        manager.metadata.create_job(job_id=track_id, kb_id=kb_id, total_files=len(files))
        parsed_texts: list[str] = []
        parsed_sources: list[str] = []
        parsed_document_ids: list[str] = []
        parsed_count = 0
        failed_count = 0
        file_payloads: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for uploaded in files:
            raw = await uploaded.read()
            filename = _safe_filename(uploaded.filename or "uploaded_file")
            extension = Path(filename).suffix.lower() or ".txt"
            document_id = _new_document_id()
            source_path = _write_source_file(kb_id, document_id, filename, raw)
            manager.metadata.create_document(
                document_id=document_id,
                kb_id=kb_id,
                original_filename=filename,
                source_file_path=source_path,
                source_file_size=len(raw),
                source_sha256=_sha256(raw),
                content_type=uploaded.content_type or "application/octet-stream",
                extension=extension,
            )

            try:
                text = parse_file_content(raw, extension)
            except ValueError as exc:
                failed_count += 1
                error_msg = str(exc)
                manager.metadata.mark_document_failed(kb_id, document_id, error_msg=error_msg)
                manager.metadata.link_job_document(
                    job_id=track_id,
                    document_id=document_id,
                    status="failed",
                    error_msg=error_msg,
                )
                document = manager.metadata.get_document(kb_id, document_id)
                file_payloads.append(document)
                errors.append({"document_id": document_id, "filename": filename, "extension": extension, "error": error_msg})
                continue

            parsed_count += 1
            manager.metadata.mark_document_parsed(kb_id, document_id, content_length=len(text))
            manager.metadata.link_job_document(job_id=track_id, document_id=document_id, status="processing")
            parsed_texts.append(text)
            parsed_sources.append(filename)
            parsed_document_ids.append(document_id)
            file_payloads.append(manager.metadata.get_document(kb_id, document_id))

        if parsed_texts:
            engine = await manager.get_engine(kb_id)
            try:
                await engine.enqueue_documents(
                    texts=parsed_texts,
                    file_sources=parsed_sources,
                    document_ids=parsed_document_ids,
                    track_id=track_id,
                )
                for document_id in parsed_document_ids:
                    manager.metadata.mark_document_indexing(kb_id, document_id)
                status = "processing"
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                for document_id in parsed_document_ids:
                    manager.metadata.update_document_index_status(
                        kb_id,
                        document_id,
                        index_status="failed",
                        error_msg=error_msg,
                    )
                    manager.metadata.link_job_document(
                        job_id=track_id,
                        document_id=document_id,
                        status="failed",
                        error_msg=error_msg,
                    )
                manager.metadata.update_job(
                    track_id,
                    status="failed",
                    parsed_count=parsed_count,
                    failed_count=failed_count + len(parsed_document_ids),
                    error_msg=error_msg,
                )
                return envelope(
                    request_id=request_id,
                    status="error",
                    data={"track_id": track_id, "files": manager.metadata.job_detail(kb_id, track_id)["documents"]},
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
        else:
            status = "failed"
        manager.metadata.update_job(
            track_id,
            status=status,
            parsed_count=parsed_count,
            failed_count=failed_count,
            error_msg="全部文件解析失败" if not parsed_texts and failed_count else None,
        )
        return envelope(
            request_id=request_id,
            data={
                "track_id": track_id,
                "kb_id": meta.kb_id,
                "kb_name": meta.name,
                "parsed_count": parsed_count,
                "error_count": failed_count,
                "total_files": len(files),
                "files": file_payloads,
                "errors": errors,
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.get("/v1/knowledge-bases/{kb_id}/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def job(kb_id: str, job_id: str) -> dict[str, Any]:
    """查询指定知识库构建任务。"""

    request_id = str(uuid.uuid4())
    try:
        try:
            engine = await manager.get_engine(kb_id)
            light_job = await engine.job(job_id)
            _sync_job_from_lightrag(kb_id, job_id, light_job)
        except Exception:
            manager.metadata.job_detail(kb_id, job_id)
        return envelope(request_id=request_id, data=manager.metadata.job_detail(kb_id, job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.get("/v1/knowledge-bases/{kb_id}/documents", dependencies=[Depends(require_api_key)])
async def documents(
    kb_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """读取指定知识库文档列表。"""

    request_id = str(uuid.uuid4())
    try:
        await _sync_documents_from_lightrag(kb_id)
        return envelope(
            request_id=request_id,
            data=manager.metadata.list_documents(kb_id, page=page, page_size=page_size),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.get("/v1/knowledge-bases/{kb_id}/documents/{document_id}", dependencies=[Depends(require_api_key)])
async def document_detail(kb_id: str, document_id: str) -> dict[str, Any]:
    """读取指定知识库内文档详情。"""

    request_id = str(uuid.uuid4())
    try:
        document = manager.metadata.get_document(kb_id, document_id)
        content = ""
        chunks: list[Any] = []
        status_raw: dict[str, Any] = {}
        light_detail = await _sync_one_document_from_lightrag(kb_id, document_id)
        if light_detail is not None:
            content = str(light_detail.get("content") or "")
            raw_chunks = light_detail.get("chunks")
            chunks = raw_chunks if isinstance(raw_chunks, list) else []
            status_payload = light_detail.get("status")
            status_raw = status_payload if isinstance(status_payload, dict) else {}
            document = manager.metadata.get_document(kb_id, document_id)
        return envelope(
            request_id=request_id,
            data={**document, "content": content, "chunks": chunks, "status_raw": status_raw},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.get("/v1/knowledge-bases/{kb_id}/documents/{document_id}/source", dependencies=[Depends(require_api_key)])
async def document_source(
    kb_id: str,
    document_id: str,
    disposition: str = Query(default="inline", pattern="^(inline|attachment)$"),
) -> FileResponse:
    """读取指定文档的原文件，用于前端预览或下载。"""

    try:
        document = manager.metadata.get_document(kb_id, document_id)
        source_path = Path(str(document.get("source_file_path") or "")).expanduser().resolve()
        allowed_dir = manager.kb_store.source_document_dir(kb_id, document_id).expanduser().resolve()
        if allowed_dir not in source_path.parents:
            raise HTTPException(status_code=403, detail="source file path is outside document source directory")
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="source file not found")
        return FileResponse(
            source_path,
            media_type=str(document.get("content_type") or "application/octet-stream"),
            filename=str(document.get("original_filename") or source_path.name),
            content_disposition_type=disposition,
        )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/v1/knowledge-bases/{kb_id}/documents/{document_id}", dependencies=[Depends(require_api_key)])
async def delete_document(
    kb_id: str,
    document_id: str,
    delete_llm_cache: bool = Query(default=False),
) -> dict[str, Any]:
    """删除指定知识库内文档、原文件和派生索引。"""

    request_id = str(uuid.uuid4())
    try:
        document = manager.metadata.get_document(kb_id, document_id)
        index_delete_error = None
        try:
            engine = await manager.get_engine(kb_id)
            await engine.delete_document(document_id, delete_llm_cache=delete_llm_cache)
        except Exception as exc:
            index_delete_error = f"{type(exc).__name__}: {exc}"
        source_dir = manager.metadata.source_document_dir(kb_id, document_id)
        shutil.rmtree(source_dir, ignore_errors=True)
        manager.metadata.delete_document_metadata(kb_id, document_id)
        return envelope(
            request_id=request_id,
            data={"deleted": True, "document": document, "index_delete_error": index_delete_error},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


@app.delete("/v1/knowledge-bases/{kb_id}/documents", dependencies=[Depends(require_api_key)])
async def clear_documents(kb_id: str) -> dict[str, Any]:
    """清空指定知识库内全部文档、原文件和索引。"""

    request_id = str(uuid.uuid4())
    try:
        meta = manager.kb_store.get(kb_id)
        engine = await manager.get_engine(kb_id)
        result = await engine.clear_documents()
        manager.metadata.clear_documents_metadata(kb_id)
        shutil.rmtree(meta.sources_dir, ignore_errors=True)
        meta.sources_dir.mkdir(parents=True, exist_ok=True)
        return envelope(request_id=request_id, data={**result, "kb_id": kb_id, "cleared_metadata": True})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return envelope(
            request_id=request_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+")


def _new_document_id() -> str:
    """生成文档 ID。"""

    return f"doc_{uuid.uuid4().hex[:16]}"


def _clean_document_id(document_id: str) -> str:
    """校验前端传入的文档 ID。"""

    clean = document_id.strip()
    if not re.match(r"^[a-zA-Z0-9_-]+$", clean):
        raise ValueError("invalid document_id")
    return clean


def _safe_filename(filename: str) -> str:
    """把上传文件名转换为安全文件名。"""

    name = Path(filename).name.strip() or "uploaded_file"
    return _FILENAME_RE.sub("_", name)[:180] or "uploaded_file"


def _write_source_file(kb_id: str, document_id: str, filename: str, raw: bytes) -> Path:
    """保存上传原文件。"""

    directory = manager.metadata.source_document_dir(kb_id, document_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(raw)
    return path


def _ensure_document_not_exists(kb_id: str, document_id: str) -> None:
    """避免指定 document_id 覆盖已有原文件。"""

    try:
        manager.metadata.get_document(kb_id, document_id)
    except KeyError:
        return
    raise ValueError(f"document already exists: {document_id}")


def _sha256(raw: bytes) -> str:
    """计算文件 SHA256。"""

    return hashlib.sha256(raw).hexdigest()


def _sync_job_from_lightrag(kb_id: str, job_id: str, light_job: dict[str, Any]) -> None:
    """用 LightRAG 任务状态同步 SQLite 元数据。"""

    for item in light_job.get("documents", []) or []:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id") or "")
        if not document_id:
            continue
        _sync_document_from_lightrag(kb_id, document_id, item)
        status = _map_lightrag_status(item)
        manager.metadata.link_job_document(
            job_id=job_id,
            document_id=document_id,
            status=status,
            error_msg=_first_error(item),
        )
    detail = manager.metadata.job_detail(kb_id, job_id)
    documents_payload = detail.get("documents", [])
    statuses = [str(item.get("job_document_status") or item.get("index_status")) for item in documents_payload]
    parsed_count = sum(1 for item in documents_payload if item.get("parse_status") == "parsed")
    failed_count = sum(1 for item in documents_payload if item.get("parse_status") == "failed" or item.get("index_status") == "failed")
    if statuses and all(status == "processed" for status in statuses):
        job_status = "processed"
    elif statuses and all(status in {"failed", "processed"} for status in statuses):
        job_status = "partial_failed" if failed_count else "processed"
    elif failed_count == detail.get("total_files"):
        job_status = "failed"
    else:
        job_status = "processing"
    manager.metadata.update_job(
        job_id,
        status=job_status,
        parsed_count=parsed_count,
        failed_count=failed_count,
    )


async def _sync_documents_from_lightrag(kb_id: str) -> None:
    """用 LightRAG doc_status 快照同步当前知识库文档索引状态。"""

    engine = await manager.get_engine(kb_id)
    page = 1
    page_size = 200
    while True:
        light_page = await engine.documents(page=page, page_size=page_size)
        documents = light_page.get("documents")
        if not isinstance(documents, list) or not documents:
            return
        for item in documents:
            if not isinstance(item, dict):
                continue
            document_id = str(item.get("document_id") or "")
            if not document_id:
                continue
            _sync_document_from_lightrag(kb_id, document_id, item)
        if not light_page.get("has_next"):
            return
        page += 1


async def _sync_one_document_from_lightrag(kb_id: str, document_id: str) -> dict[str, Any] | None:
    """同步单个文档；文档还没进入 LightRAG 时返回 None。"""

    engine = await manager.get_engine(kb_id)
    try:
        light_detail = await engine.document_detail(document_id)
    except KeyError:
        return None
    status_payload = light_detail.get("status")
    if isinstance(status_payload, dict):
        _sync_document_from_lightrag(kb_id, document_id, status_payload)
    return light_detail


def _sync_document_from_lightrag(kb_id: str, document_id: str, status_payload: dict[str, Any]) -> None:
    """同步单个 LightRAG 文档状态。"""

    status = _map_lightrag_status(status_payload)
    chunks_count = status_payload.get("chunks_count") or status_payload.get("chunk_count")
    manager.metadata.update_document_index_status(
        kb_id,
        document_id,
        index_status=status,
        chunks_count=int(chunks_count or 0),
        error_msg=_first_error(status_payload),
    )


def _map_lightrag_status(payload: dict[str, Any]) -> str:
    """把 LightRAG 状态映射到产品状态。"""

    raw = str(payload.get("status") or payload.get("doc_status") or "processing").lower()
    if raw in {"processed", "done", "success", "completed"}:
        return "processed"
    if raw in {"failed", "error"}:
        return "failed"
    return "processing"


def _first_error(payload: dict[str, Any]) -> str | None:
    """从 LightRAG 状态中提取错误信息。"""

    for key in ("error", "error_msg", "message"):
        value = payload.get(key)
        if value:
            return str(value)
    return None
