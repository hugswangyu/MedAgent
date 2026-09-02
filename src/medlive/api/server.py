"""前端管理接口。"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from medlive.api.rag_gateway import GatewayResponse, RagGateway, envelope
from medlive.config.settings import (
    RagClientSettings,
    RagToolMode,
    is_masked_secret,
    load_app_settings,
    load_context_model_settings,
    load_environment,
    load_rag_client_settings,
    load_voice_settings,
    public_context_model_config,
    public_model_options,
    public_rag_client_config,
    public_voice_config,
    read_runtime_context_model_config,
    read_runtime_model_config,
    validate_voice_config_selection,
    voice_config_for_storage,
    write_runtime_context_model_config,
    write_runtime_model_config,
)
from medlive.context.overview import KnowledgeOverviewGenerator
from medlive.context.store import ContextStore
from medlive.control_plane import ControlPlaneClient, ControlPlaneError
from medlive.rag.metadata_store import MetadataStore
from medlive.rag.schemas import QueryRequest, TextDocumentRequest
from medlive.rag.service import wait_for_rag_ready
from medlive.runtime.paths import build_runtime_paths
from medlive.security import CurrentUser, get_current_admin, get_current_user
from medlive.voice.session import VoiceSessionError, VoiceSessionService

load_environment()
settings = load_app_settings()
paths = build_runtime_paths(settings.user_data_dir)
store = ContextStore(paths)
store.initialize(reset_session=False)
metadata_store = MetadataStore(paths.db_file, paths.rag_knowledge_bases_dir)
metadata_store.initialize()
control_plane = ControlPlaneClient()
rag_gateway = RagGateway(settings)
app = FastAPI(title="LiveRAG Agent API", version="0.1.0")
UPLOAD_FILES = File(...)


class TextPayload(BaseModel):
    """文本更新请求。"""

    content: str


class RagConfigPayload(BaseModel):
    """RAG 配置更新请求。"""

    enabled: bool | None = None
    base_url: str | None = None
    api_key: str | None = None
    query_mode: str | None = None
    timeout_ms: int | None = None
    top_k: int | None = None
    chunk_top_k: int | None = None
    context_max_chars: int | None = None
    cache_ttl_s: float | None = None
    enable_rerank: bool | None = None
    rag_tool_mode: RagToolMode | None = None


class KnowledgeBasePayload(BaseModel):
    """知识库创建或更新请求。"""

    name: str | None = None
    description: str | None = None


class SessionKnowledgeBasePayload(BaseModel):
    """会话知识库选择请求。"""

    kb_id: str


class VoiceSessionCreatePayload(BaseModel):
    """创建业务语音会话请求。"""

    kb_id: str
    client_id: str | None = None
    client_type: Literal["android", "web", "test"] = "android"


class ModelSttPayload(BaseModel):
    """语音 STT 模型配置更新请求。"""

    provider: str | None = None
    model: str | None = None
    app_id: str | None = None
    access_token: str | None = None


class ModelLlmPayload(BaseModel):
    """语音 LLM 模型配置更新请求。"""

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class ModelTtsPayload(BaseModel):
    """语音 TTS 模型配置更新请求。"""

    provider: str | None = None
    model: str | None = None
    voice: str | None = None
    api_key: str | None = None


class ModelVoicePayload(BaseModel):
    """语音模型配置更新请求。"""

    stt: ModelSttPayload | None = None
    llm: ModelLlmPayload | None = None
    tts: ModelTtsPayload | None = None


class ModelConfigPayload(BaseModel):
    """模型配置更新请求。"""

    voice: ModelVoicePayload | None = None


class ContextModelPayload(BaseModel):
    """上下文模型配置更新请求。"""

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_session_chars: int | None = None
    history_reference_limit: int | None = None
    timeout_ms: int | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    """返回管理接口健康状态。"""

    return {"status": "ok"}


@app.get("/runtime/state")
async def runtime_state(
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """读取当前运行状态。"""

    state = store.read_runtime_state()
    active_session = state.get("active_session")
    if (
        isinstance(active_session, dict)
        and active_session.get("user_id") != current_user.user_id
    ):
        state.pop("active_session", None)
        state.pop("active_voice_model", None)
        active_session = None
    state.setdefault("rag_tool_mode", load_rag_client_settings(settings.user_data_dir).rag_tool_mode)
    if isinstance(active_session, dict):
        state.setdefault("active_voice_model", active_session.get("voice"))
        state.setdefault("model_pending_reconnect", _model_pending_reconnect(active_session.get("voice")))
    state["knowledge_base"] = await _session_knowledge_base_state(
        current_user.user_id, current_user.access_token
    )
    return state


@app.get("/model/config")
async def model_config(
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, Any]:
    """读取下次通话将使用的语音模型配置。"""

    voice = load_voice_settings(settings.user_data_dir)
    return envelope(
        data={
            "voice": public_voice_config(voice, effective="next_session"),
        }
    )


@app.get("/model/options")
async def model_options() -> dict[str, Any]:
    """读取前端模型选择页可用 provider、模型和音色。"""

    return envelope(data=public_model_options())


@app.put("/model/config")
async def put_model_config(
    payload: ModelConfigPayload,
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, Any]:
    """更新语音模型运行时配置。"""

    updates = _drop_masked_secret_updates(payload.model_dump(exclude_none=True))
    _validate_model_config_updates(updates)
    current = read_runtime_model_config(settings.user_data_dir)
    merged = voice_config_for_storage(_deep_merge_dict(current, updates))
    try:
        validate_voice_config_selection(merged)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    write_runtime_model_config(merged, settings.user_data_dir)
    voice = load_voice_settings(settings.user_data_dir)
    return envelope(
        data={
            "voice": public_voice_config(voice, effective="next_session"),
        }
    )


@app.get("/model/effective-state")
async def model_effective_state(
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, Any]:
    """读取模型配置和当前/最近通话实际生效状态。"""

    configured = {
        "voice": public_voice_config(load_voice_settings(settings.user_data_dir), effective="next_session"),
        "options": public_model_options(),
    }
    active_session = store.read_runtime_state().get("active_session")
    if not isinstance(active_session, dict):
        active_session = None
    return envelope(
        data={
            "configured": configured,
            "active_session": active_session,
            "pending_reconnect": _model_pending_reconnect(
                active_session.get("voice") if active_session else None
            ),
        }
    )


@app.get("/model/context-config")
async def context_model_config(
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, Any]:
    """读取上下文模型配置。"""

    config = load_context_model_settings(settings.user_data_dir)
    return envelope(data={"context_model": public_context_model_config(config)})


@app.put("/model/context-config")
async def put_context_model_config(
    payload: ContextModelPayload,
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, Any]:
    """更新上下文模型运行时配置。"""

    updates = _drop_masked_secret_updates(payload.model_dump(exclude_none=True))
    _validate_context_model_updates(updates)
    current = read_runtime_context_model_config(settings.user_data_dir)
    merged = {**current, **updates}
    write_runtime_context_model_config(merged, settings.user_data_dir)
    config = load_context_model_settings(settings.user_data_dir)
    return envelope(data={"context_model": public_context_model_config(config)})


@app.get("/prompt/soul")
async def get_soul(
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, str]:
    """读取用户定义的 Agent 角色人格。"""

    return {"content": store.read_soul()}


@app.put("/prompt/soul")
async def put_soul(
    payload: TextPayload,
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, str]:
    """更新用户定义的 Agent 角色人格。"""

    store.write_soul(payload.content)
    return {"status": "ok"}


@app.get("/session/messages")
async def session_messages(
    limit: int | None = None,
    _admin: CurrentUser = Depends(get_current_admin),
) -> list[dict[str, Any]]:
    """读取当前会话消息。"""

    return store.read_messages(limit=limit)


@app.get("/session/rag-context")
async def session_rag_context(
    limit: int | None = None,
    _admin: CurrentUser = Depends(get_current_admin),
) -> list[dict[str, Any]]:
    """读取当前会话 RAG 查询记录。"""

    return store.read_rag_context(limit=limit)


@app.get("/session/turns")
async def session_turns(
    limit: int | None = None,
    _admin: CurrentUser = Depends(get_current_admin),
) -> list[dict[str, Any]]:
    """按轮次读取当前会话消息和 RAG 展示依据。"""

    return store.read_session_turns(limit=limit)


@app.post("/session/clear")
async def clear_session(
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, str]:
    """清空当前单会话数据。"""

    store.clear_session()
    return {"status": "ok"}


@app.post("/voice/sessions")
async def create_voice_session(
    payload: VoiceSessionCreatePayload,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """创建 Android/Web/Test 业务语音会话并签发 LiveKit token。"""

    service = _voice_session_service()
    try:
        data = await service.create_session(
            user_id=current_user.user_id,
            kb_id=payload.kb_id,
            client_id=payload.client_id,
            client_type=payload.client_type,
            access_token=current_user.access_token,
        )
    except VoiceSessionError as exc:
        return _voice_session_error_response(exc)
    return JSONResponse(envelope(data=data), status_code=200)


@app.get("/voice/sessions/{session_id}")
async def get_voice_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """读取业务语音会话状态。"""

    service = _voice_session_service()
    try:
        data = await service.get_session(
            session_id,
            user_id=current_user.user_id,
            access_token=current_user.access_token,
        )
    except KeyError:
        return _error_response(404, "VoiceSessionNotFound", f"voice session not found: {session_id}")
    except VoiceSessionError as exc:
        return _voice_session_error_response(exc)
    return JSONResponse(envelope(data=data), status_code=200)


@app.post("/voice/sessions/{session_id}/token")
async def refresh_voice_session_token(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """刷新指定 voice session 的 LiveKit join token。"""

    service = _voice_session_service()
    try:
        data = await service.refresh_token(
            session_id,
            user_id=current_user.user_id,
            access_token=current_user.access_token,
        )
    except KeyError:
        return _error_response(404, "VoiceSessionNotFound", f"voice session not found: {session_id}")
    except VoiceSessionError as exc:
        return _voice_session_error_response(exc)
    return JSONResponse(envelope(data=data), status_code=200)


@app.get("/voice/sessions/{session_id}/turns")
async def get_voice_session_turns(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int | None = Query(default=None, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """读取指定 voice session 的 turns。"""

    service = _voice_session_service()
    try:
        data = await service.paginate_turns(
            session_id,
            user_id=current_user.user_id,
            limit=limit,
            cursor=cursor,
            access_token=current_user.access_token,
        )
    except KeyError:
        return _error_response(404, "VoiceSessionNotFound", f"voice session not found: {session_id}")
    return JSONResponse(envelope(data=data), status_code=200)


@app.get("/voice/sessions/{session_id}/rag-context")
async def get_voice_session_rag_context(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int | None = Query(default=None, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """读取指定 voice session 的 RAG context。"""

    service = _voice_session_service()
    try:
        data = await service.paginate_rag_context(
            session_id,
            user_id=current_user.user_id,
            limit=limit,
            cursor=cursor,
            access_token=current_user.access_token,
        )
    except KeyError:
        return _error_response(404, "VoiceSessionNotFound", f"voice session not found: {session_id}")
    return JSONResponse(envelope(data=data), status_code=200)


@app.post("/voice/sessions/{session_id}/end")
async def end_voice_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """幂等结束业务语音会话。"""

    service = _voice_session_service()
    try:
        data = await service.end_session(
            session_id,
            reason="api",
            user_id=current_user.user_id,
            access_token=current_user.access_token,
        )
    except KeyError:
        return _error_response(404, "VoiceSessionNotFound", f"voice session not found: {session_id}")
    except VoiceSessionError as exc:
        return _voice_session_error_response(exc)
    return JSONResponse(envelope(data=data), status_code=200)


@app.get("/session/knowledge-base")
async def get_session_knowledge_base(
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """读取下次通话配置和当前通话锁定的知识库。"""

    return await _session_knowledge_base_state(
        current_user.user_id, current_user.access_token
    )


@app.put("/session/knowledge-base")
async def put_session_knowledge_base(
    payload: SessionKnowledgeBasePayload,
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """设置下次通话使用的知识库。"""

    locked = _active_knowledge_base(current_user.user_id)
    if locked is not None:
        raise HTTPException(status_code=409, detail="KnowledgeBaseLocked")
    await _require_owned_knowledge_base(payload.kb_id, current_user.access_token)
    kb = await _knowledge_base_detail(payload.kb_id)
    metadata_store.set_session_config(
        f"knowledge_base:{current_user.user_id}",
        {"kb_id": kb["kb_id"], "name": kb["name"]},
    )
    await rag_gateway.get(f"/v1/knowledge-bases/{kb['kb_id']}/ready")
    return await _session_knowledge_base_state(
        current_user.user_id, current_user.access_token
    )


@app.get("/rag/config")
async def rag_config(
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, Any]:
    """读取语音链路 RAG 配置。"""

    config = load_rag_client_settings(settings.user_data_dir)
    return envelope(data={"config": public_rag_client_config(config)})


@app.put("/rag/config")
async def put_rag_config(
    payload: RagConfigPayload,
    _admin: CurrentUser = Depends(get_current_admin),
) -> dict[str, Any]:
    """更新语音链路 RAG 配置。"""

    current = load_rag_client_settings(settings.user_data_dir).__dict__
    updates = payload.model_dump(exclude_none=True)
    if is_masked_secret(updates.get("api_key")):
        updates.pop("api_key", None)
    merged = {**current, **updates}
    try:
        validated = RagClientSettings(**merged)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    config_path = paths.rag_dir / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(validated.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return envelope(data={"config": public_rag_client_config(validated)})


@app.get("/rag/knowledge-bases")
async def rag_knowledge_bases(
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """返回全部知识库。"""

    ownership = await control_plane.list_knowledge_bases(current_user.access_token)
    items = []
    for owned in ownership.get("knowledge_bases", []):
        try:
            items.append(metadata_store.public_knowledge_base_detail(str(owned["kb_id"])))
        except KeyError:
            continue
    return JSONResponse(
        envelope(data={"knowledge_bases": items, "total": len(items)}),
        status_code=200,
    )


@app.post("/rag/knowledge-bases")
async def rag_create_knowledge_base(
    payload: KnowledgeBasePayload,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """创建知识库。"""

    result = await rag_gateway.post_json(
        "/v1/knowledge-bases",
        payload=payload.model_dump(exclude_none=True),
    )
    if result.status_code < 400 and isinstance(result.body.get("data"), dict):
        kb_id = str(result.body["data"].get("kb_id") or "")
        if kb_id:
            try:
                await control_plane.register_knowledge_base(current_user.access_token, kb_id)
            except ControlPlaneError as exc:
                await rag_gateway.delete(f"/v1/knowledge-bases/{kb_id}")
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _json_response(result)


@app.get("/rag/knowledge-bases/{kb_id}")
async def rag_knowledge_base_detail(
    kb_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """返回单个知识库详情。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    return _json_response(await rag_gateway.get(f"/v1/knowledge-bases/{kb_id}"))


@app.patch("/rag/knowledge-bases/{kb_id}")
async def rag_patch_knowledge_base(
    kb_id: str,
    payload: KnowledgeBasePayload,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """更新知识库元数据。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    return _json_response(
        await rag_gateway.patch_json(
            f"/v1/knowledge-bases/{kb_id}",
            payload=payload.model_dump(exclude_none=True),
        )
    )


@app.delete("/rag/knowledge-bases/{kb_id}")
async def rag_delete_knowledge_base(
    kb_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """删除知识库。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    active = _active_knowledge_base(current_user.user_id)
    if active and active.get("kb_id") == kb_id:
        return JSONResponse(
            envelope(
                status="error",
                error={"type": "KnowledgeBaseLocked", "message": "当前通话正在使用该知识库"},
            ),
            status_code=409,
        )
    await control_plane.set_knowledge_base_status(
        current_user.access_token, kb_id, "deleted"
    )
    result = await rag_gateway.delete(f"/v1/knowledge-bases/{kb_id}")
    if result.body.get("status") != "ok":
        await control_plane.set_knowledge_base_status(
            current_user.access_token, kb_id, "active"
        )
    config_key = f"knowledge_base:{current_user.user_id}"
    configured = metadata_store.get_session_config(config_key)
    if result.body.get("status") == "ok" and configured.get("kb_id") == kb_id:
        metadata_store.set_session_config(config_key, {})
    return _json_response(result)


@app.get("/rag/knowledge-bases/{kb_id}/ready")
async def rag_knowledge_base_ready(
    kb_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """预热知识库 engine。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    return _json_response(await rag_gateway.get(f"/v1/knowledge-bases/{kb_id}/ready"))


@app.get("/rag/knowledge-bases/{kb_id}/context/overview")
async def rag_kb_context_overview(
    kb_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """读取指定知识库的固定上下文概览。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    await _knowledge_base_detail(kb_id)
    return envelope(
        data={
            "kb_id": kb_id,
            "content": store.read_knowledge_overview(kb_id),
            "meta": store.read_knowledge_overview_meta(kb_id),
        }
    )


@app.put("/rag/knowledge-bases/{kb_id}/context/overview")
async def put_rag_kb_context_overview(
    kb_id: str,
    payload: TextPayload,
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """手动覆盖指定知识库的固定上下文概览。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    kb = await _knowledge_base_detail(kb_id)
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="content cannot be empty")
    store.write_knowledge_overview(
        kb_id,
        content,
        stale=False,
        reason="manual_update",
        source="manual",
    )
    return envelope(
        data={
            "kb_id": kb["kb_id"],
            "kb_name": kb["name"],
            "content": store.read_knowledge_overview(kb_id),
            "meta": store.read_knowledge_overview_meta(kb_id),
        }
    )


@app.get("/rag/knowledge-bases/{kb_id}/documents")
async def rag_kb_documents(
    kb_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """返回指定知识库文档列表。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    return _json_response(
        await rag_gateway.get_documents(
            f"/v1/knowledge-bases/{kb_id}/documents",
            params={"page": page, "page_size": page_size},
        )
    )


@app.get("/rag/knowledge-bases/{kb_id}/documents/{document_id}")
async def rag_kb_document_detail(
    kb_id: str,
    document_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """返回指定知识库文档详情。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    return _json_response(
        await rag_gateway.get_document_detail(f"/v1/knowledge-bases/{kb_id}/documents/{document_id}")
    )


@app.get("/rag/knowledge-bases/{kb_id}/documents/{document_id}/source")
async def rag_kb_document_source(
    kb_id: str,
    document_id: str,
    disposition: str = Query(default="inline", pattern="^(inline|attachment)$"),
    current_user: CurrentUser = Depends(get_current_user),
) -> Response:
    """返回指定知识库文档原文件，用于前端预览或下载。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    result = await rag_gateway.get_file(
        f"/v1/knowledge-bases/{kb_id}/documents/{document_id}/source",
        params={"disposition": disposition},
    )
    if result.error_body is not None:
        return JSONResponse(result.error_body, status_code=result.status_code)
    return Response(content=result.body, status_code=result.status_code, headers=result.headers)


@app.post("/rag/knowledge-bases/{kb_id}/documents/text")
async def rag_kb_documents_text(
    kb_id: str,
    payload: TextDocumentRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """向指定知识库导入文本。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    result = await rag_gateway.post_json(
        f"/v1/knowledge-bases/{kb_id}/documents/text",
        payload=payload.model_dump(exclude_none=True),
    )
    _mark_overview_stale_if_ok(result, kb_id, reason="documents_text_imported")
    return _json_response(result)


@app.post("/rag/knowledge-bases/{kb_id}/documents/files")
async def rag_kb_documents_files(
    kb_id: str,
    files: list[UploadFile] = UPLOAD_FILES,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """向指定知识库上传文件。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    result = await rag_gateway.post_files(f"/v1/knowledge-bases/{kb_id}/documents/files", files=files)
    _mark_overview_stale_if_ok(result, kb_id, reason="documents_files_uploaded")
    return _json_response(result)


@app.get("/rag/knowledge-bases/{kb_id}/jobs/{job_id}")
async def rag_kb_job(
    kb_id: str,
    job_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """查询指定知识库构建任务。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    result = await rag_gateway.get_job(f"/v1/knowledge-bases/{kb_id}/jobs/{job_id}")
    _schedule_overview_generation_after_completed_job(
        result,
        kb_id=kb_id,
        job_id=job_id,
        background_tasks=background_tasks,
    )
    return _json_response(result)


@app.delete("/rag/knowledge-bases/{kb_id}/documents/{document_id}")
async def rag_kb_delete_document(
    kb_id: str,
    document_id: str,
    delete_llm_cache: bool = Query(default=False),
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """删除指定知识库文档。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    result = await rag_gateway.delete(
        f"/v1/knowledge-bases/{kb_id}/documents/{document_id}",
        params={"delete_llm_cache": delete_llm_cache},
    )
    _mark_overview_stale_if_ok(result, kb_id, reason="document_deleted")
    return _json_response(result)


@app.delete("/rag/knowledge-bases/{kb_id}/documents")
async def rag_kb_clear_documents(
    kb_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """清空指定知识库文档。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    result = await rag_gateway.delete(f"/v1/knowledge-bases/{kb_id}/documents")
    _mark_overview_stale_if_ok(result, kb_id, reason="documents_cleared")
    return _json_response(result)


@app.post("/rag/knowledge-bases/{kb_id}/query/context")
async def rag_kb_query_context(
    kb_id: str,
    payload: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """查询指定知识库上下文。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    return _json_response(
        await rag_gateway.post_json(
            f"/v1/knowledge-bases/{kb_id}/query/context",
            payload=payload.model_dump(exclude_none=True),
        )
    )


@app.post("/rag/knowledge-bases/{kb_id}/query/data")
async def rag_kb_query_data(
    kb_id: str,
    payload: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """查询指定知识库结构化数据。"""

    await _require_owned_knowledge_base(kb_id, current_user.access_token)
    return _json_response(
        await rag_gateway.post_json(
            f"/v1/knowledge-bases/{kb_id}/query/data",
            payload=payload.model_dump(exclude_none=True),
        )
    )


@app.post("/rag/session-query/context")
async def rag_session_query_context(
    payload: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """按当前会话锁定知识库查询上下文。"""

    kb = await _effective_session_knowledge_base(
        current_user.user_id, current_user.access_token
    )
    return await rag_kb_query_context(kb["kb_id"], payload, current_user)


@app.post("/rag/session-query/data")
async def rag_session_query_data(
    payload: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """按当前会话锁定知识库查询结构化数据。"""

    kb = await _effective_session_knowledge_base(
        current_user.user_id, current_user.access_token
    )
    return await rag_kb_query_data(kb["kb_id"], payload, current_user)


@app.get("/rag/ready")
async def rag_ready() -> JSONResponse:
    """返回内部 RAG 服务 ready 状态。"""

    return _json_response(await rag_gateway.get("/v1/readyz"))


@app.on_event("startup")
async def startup_event() -> None:
    """管理 API 启动时确保内部 RAG 服务可用。"""

    await _voice_session_service().cleanup_stale_sessions()
    app.state.voice_session_cleanup_task = asyncio.create_task(_voice_session_cleanup_loop())
    await asyncio.to_thread(wait_for_rag_ready, timeout_ms=settings.api.rag_ready_timeout_ms)


def _json_response(result: GatewayResponse) -> JSONResponse:
    """把网关返回结构转换成 FastAPI 响应。"""

    return JSONResponse(result.body, status_code=result.status_code)


def _voice_session_service() -> VoiceSessionService:
    """构造最新配置下的 voice session service。"""

    current_settings = load_app_settings()
    return VoiceSessionService(
        settings=current_settings,
        store=store,
        metadata_store=metadata_store,
        control_plane=control_plane,
    )


async def _require_owned_knowledge_base(kb_id: str, access_token: str) -> None:
    """不存在与不属于当前用户统一返回 404，避免枚举他人知识库。"""

    try:
        await control_plane.get_knowledge_base(access_token, kb_id)
    except ControlPlaneError as exc:
        raise HTTPException(status_code=404, detail="knowledge base not found") from exc


def _voice_session_error_response(exc: VoiceSessionError) -> JSONResponse:
    """把 voice session 领域错误转换成统一 envelope。"""

    status_code, body = exc.to_response()
    return JSONResponse(body, status_code=status_code)


def _error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    """返回统一错误响应。"""

    return JSONResponse(
        envelope(status="error", data=None, error={"type": error_type, "message": message}),
        status_code=status_code,
    )


async def _voice_session_cleanup_loop() -> None:
    """周期清理过期 voice session。"""

    while True:
        await asyncio.sleep(max(load_app_settings().api.voice_session_cleanup_interval_s, 10))
        try:
            await _voice_session_service().cleanup_stale_sessions()
        except Exception:
            continue


def _active_knowledge_base(user_id: str | None = None) -> dict[str, Any] | None:
    """返回当前未结束通话锁定的知识库。"""

    state = store.read_runtime_state()
    active_session = state.get("active_session")
    if not isinstance(active_session, dict) or active_session.get("ended_at"):
        return None
    if user_id is not None and active_session.get("user_id") != user_id:
        return None
    knowledge_base = active_session.get("knowledge_base")
    return knowledge_base if isinstance(knowledge_base, dict) else None


async def _configured_knowledge_base(
    user_id: str, access_token: str
) -> dict[str, Any]:
    """读取当前用户下次通话知识库，缺失时选择其首个自有库。"""

    configured = metadata_store.get_session_config(f"knowledge_base:{user_id}")
    kb_id = str(configured.get("kb_id") or "")
    if kb_id:
        try:
            await _require_owned_knowledge_base(kb_id, access_token)
            return await _knowledge_base_detail(kb_id)
        except HTTPException:
            pass
    owned = (
        await control_plane.list_knowledge_bases(access_token)
    ).get("knowledge_bases", [])
    if not owned:
        raise HTTPException(status_code=404, detail="current user has no knowledge base")
    return await _knowledge_base_detail(str(owned[0]["kb_id"]))


async def _effective_session_knowledge_base(
    user_id: str, access_token: str
) -> dict[str, Any]:
    """返回查询实际应使用的知识库。"""

    active = _active_knowledge_base(user_id)
    if active and active.get("kb_id"):
        return {"kb_id": str(active["kb_id"]), "name": str(active.get("name") or active["kb_id"])}
    return await _configured_knowledge_base(user_id, access_token)


async def _session_knowledge_base_state(
    user_id: str, access_token: str
) -> dict[str, Any]:
    """返回会话知识库配置和锁定状态。"""

    configured = await _configured_knowledge_base(user_id, access_token)
    active = _active_knowledge_base(user_id)
    return {
        "configured": {"kb_id": configured["kb_id"], "name": configured["name"]},
        "active_session": active,
        "locked": active is not None,
        "pending_reconnect": bool(active and active.get("kb_id") != configured["kb_id"]),
    }


async def _knowledge_base_detail(kb_id: str) -> dict[str, Any]:
    """通过内部 RAG 服务读取知识库详情。"""

    result = await rag_gateway.get(f"/v1/knowledge-bases/{kb_id}")
    if result.status_code == 404:
        raise HTTPException(status_code=404, detail=f"knowledge base not found: {kb_id}")
    if result.body.get("status") != "ok" or not isinstance(result.body.get("data"), dict):
        error = result.body.get("error") or {}
        raise HTTPException(status_code=result.status_code, detail=error.get("message") or "knowledge base unavailable")
    return result.body["data"]


async def _raw_knowledge_overview(kb_id: str) -> dict[str, Any] | None:
    """读取内部 RAG Core 的结构化知识库概览。"""

    result = await rag_gateway.get(
        f"/v1/knowledge-bases/{kb_id}/overview",
        params={
            "entity_limit": 20,
            "relation_limit": 12,
            "document_limit": 20,
            "topic_limit": 12,
        },
    )
    if result.body.get("status") != "ok":
        return None
    data = result.body.get("data")
    return data if isinstance(data, dict) else None


def _schedule_overview_generation_after_completed_job(
    result: GatewayResponse,
    *,
    kb_id: str,
    job_id: str,
    background_tasks: BackgroundTasks,
) -> None:
    """索引任务完成且有新文档构建成功时，安排后台生成知识库概览。"""

    if result.body.get("status") != "ok":
        return
    data = result.body.get("data")
    if not isinstance(data, dict) or not _job_has_new_processed_documents(data):
        return
    meta = store.read_knowledge_overview_meta(kb_id)
    if meta.get("source_job_id") == job_id and not meta.get("stale"):
        return
    background_tasks.add_task(_generate_overview_for_completed_job, kb_id, job_id)
    data["overview_generation"] = {"scheduled": True, "trigger": "index_completed", "job_id": job_id}


def _job_has_new_processed_documents(data: dict[str, Any]) -> bool:
    """判断任务是否已结束且至少有一个文档构建成功。"""

    status = str(data.get("status") or "").lower()
    if status not in {"processed", "partial_failed"}:
        return False
    documents = data.get("documents")
    if not isinstance(documents, list):
        return False
    for document in documents:
        if not isinstance(document, dict):
            continue
        index_status = str(document.get("index_status") or document.get("job_document_status") or "").lower()
        if index_status == "processed":
            return True
    return False


async def _generate_overview_for_completed_job(kb_id: str, job_id: str) -> None:
    """为已完成索引任务生成知识库概览。"""

    try:
        kb = await _knowledge_base_detail(kb_id)
        raw_overview = await _raw_knowledge_overview(kb_id)
        await KnowledgeOverviewGenerator(
            store=store,
            settings=load_context_model_settings(settings.user_data_dir),
        ).generate(
            kb_id=kb["kb_id"],
            kb_name=kb["name"],
            raw_overview=raw_overview,
            rag_settings=load_rag_client_settings(settings.user_data_dir),
            reason="index_completed",
            source_job_id=job_id,
        )
    except Exception:
        # 后台概览生成失败不影响任务查询接口；具体失败会在 generator 内部写入 meta。
        return


def _mark_overview_stale_if_ok(result: GatewayResponse, kb_id: str, *, reason: str) -> None:
    """文档变更成功后标记对应知识库概览过期。"""

    if result.body.get("status") == "ok":
        store.mark_knowledge_overview_stale(kb_id, reason=reason)


def _validate_model_config_updates(updates: dict[str, Any]) -> None:
    """校验模型配置局部更新。"""

    for path, value in _walk_model_update_values(updates):
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=422, detail=f"{path} cannot be empty")
        if path == "voice.stt.provider" and value.strip().lower() != "volcengine_bigmodel":
            raise HTTPException(status_code=422, detail="voice.stt.provider must be volcengine_bigmodel")
        if path == "voice.tts.provider" and value.strip().lower() not in {"minimax", "dashscope_realtime"}:
            raise HTTPException(status_code=422, detail="voice.tts.provider must be minimax or dashscope_realtime")
        if path.endswith(".base_url") and not _is_http_url(value):
            raise HTTPException(status_code=422, detail=f"{path} must be an http(s) URL")


def _validate_context_model_updates(updates: dict[str, Any]) -> None:
    """校验上下文模型配置局部更新。"""

    for key, value in updates.items():
        if key in {"model", "base_url", "api_key"}:
            if not isinstance(value, str) or not value.strip():
                raise HTTPException(status_code=422, detail=f"{key} cannot be empty")
            if key == "base_url" and not _is_http_url(value):
                raise HTTPException(status_code=422, detail="base_url must be an http(s) URL")
        elif key in {"max_tokens", "max_session_chars", "history_reference_limit", "timeout_ms"}:
            if not isinstance(value, int) or value <= 0:
                raise HTTPException(status_code=422, detail=f"{key} must be a positive integer")
        elif key == "temperature" and not isinstance(value, (int, float)):
            raise HTTPException(status_code=422, detail="temperature must be a number")


def _drop_masked_secret_updates(payload: dict[str, Any]) -> dict[str, Any]:
    """移除前端原样回传的密钥掩码，避免覆盖真实密钥。"""

    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            nested = _drop_masked_secret_updates(value)
            if nested:
                cleaned[key] = nested
            continue
        if key in {"api_key", "access_token"} and is_masked_secret(value):
            continue
        cleaned[key] = value
    return cleaned


def _walk_model_update_values(payload: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """展开模型配置更新字段。"""

    items: list[tuple[str, Any]] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_walk_model_update_values(value, path))
        else:
            items.append((path, value))
    return items


def _deep_merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典。"""

    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _is_http_url(value: str) -> bool:
    """判断字符串是否是 http(s) URL。"""

    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _model_pending_reconnect(active_voice: Any) -> bool:
    """判断当前配置是否还未通过重连生效。"""

    if not isinstance(active_voice, dict):
        return False
    configured = public_voice_config(load_voice_settings(settings.user_data_dir), effective="next_session")
    return _voice_config_identity(configured) != _voice_config_identity(active_voice)


def _voice_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """去掉展示型 effective 字段后比较语音模型配置。"""

    identity: dict[str, Any] = {}
    for section, values in config.items():
        if isinstance(values, dict):
            identity[section] = {key: value for key, value in values.items() if key != "effective"}
    return identity
