"""SSE 流式聊天端点：POST /chat/stream、GET /models。"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import StreamingResponse

from medrag.config.settings import settings

from ..dependencies import get_current_user
from ..schemas import ChatRequest, ModelItem, ModelsResponse
from ..session_store import add_message

logger = logging.getLogger(__name__)

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4)

# 延迟导入以在 lifespan 启动完成后才解析
_chat_service = None


def set_chat_service(service):
    global _chat_service
    _chat_service = service


def _get_service():
    if _chat_service is None:
        raise HTTPException(status_code=503, detail="聊天服务尚未就绪")
    return _chat_service


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """返回已配置 API Key 的可用提供商及模型列表。"""
    providers: list[ModelItem] = []
    if settings.deepseek_api_key:
        providers.append(ModelItem(
            provider="deepseek",
            models=list(settings.deepseek_model_options),
        ))
    if settings.zhipuai_api_key:
        providers.append(ModelItem(
            provider="zhipuai",
            models=list(settings.zhipuai_model_options),
        ))
    if settings.qwen_api_key:
        providers.append(ModelItem(
            provider="qwen",
            models=list(settings.qwen_model_options),
        ))
    providers.append(ModelItem(
        provider="ollama",
        models=list(settings.ollama_model_options),
    ))
    return ModelsResponse(providers=providers)


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    current_user=Depends(get_current_user),
):
    service = _get_service()
    department = body.knowledge_base if body.knowledge_base != "全科" else None

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            for event in service.stream_chat(
                query=body.message,
                username=current_user.username,
                department=department,
                provider=body.provider,
                model=body.model,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
        except Exception as exc:
            logger.error("stream_chat failed: %s", exc, exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    loop.run_in_executor(_executor, _run)

    async def event_generator():
        collected_content = ""
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    yield f"data: {json.dumps({'type': 'error', 'content': str(event)})}\n\n"
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] == "content":
                    collected_content += event["content"]
        finally:
            yield "data: [DONE]\n\n"
            # 存入会话（带用户归属）
            try:
                add_message(body.session_id, "human", body.message,
                            username=current_user.username)
                add_message(body.session_id, "ai", collected_content,
                            username=current_user.username)
            except Exception:
                logger.warning("保存会话消息失败", exc_info=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
