"""FastAPI 应用入口。

启动::

    uvicorn medrag.app.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import (
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .api import auth, chat, control_v1, documents, internal_v1, memories, sessions
from .auth_manager import init_auth

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

_chat_service = None
_chat_service_error: str | None = None

# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chat_service, _chat_service_error

    logger.info("正在初始化认证系统...")
    init_auth()

    logger.info("正在加载 MedicalChatService（可能需要一些时间）...")
    try:
        from medrag.service.chat_service import MedicalChatService as Svc
        _chat_service = Svc()
        logger.info("MedicalChatService 加载完成")
    except Exception as exc:
        _chat_service_error = str(exc)
        _chat_service = None
        logger.warning("MedicalChatService 加载失败（聊天功能不可用）: %s", exc)

    chat.set_chat_service(_chat_service)
    internal_v1.bind_chat_service(_chat_service)

    yield

    logger.info("服务器关闭")


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"

app = FastAPI(
    title="MedAgent — Medical AI Agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由（在静态文件之前注册）
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(memories.router, prefix="/memories", tags=["memories"])
app.include_router(control_v1.router, prefix="/control/v1", tags=["control-plane"])
app.include_router(
    internal_v1.router,
    prefix="/internal/v1",
    tags=["internal-capabilities"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """internal v1 验证错误使用冻结 envelope，其余 API 保持 FastAPI 默认。"""

    if not request.url.path.startswith("/internal/v1/"):
        return await request_validation_exception_handler(request, exc)
    envelope = internal_v1.capabilities.error(
        "INVALID_REQUEST",
        "请求参数不符合阶段 0 契约",
        request_id=f"req_{uuid.uuid4().hex}",
        details={"errors": jsonable_encoder(exc.errors())},
    )
    return JSONResponse(
        status_code=422,
        content=envelope.model_dump(mode="json"),
    )


@app.get("/health")
async def health():
    from medrag.infrastructure.health import get_summary
    summary = get_summary()
    if _chat_service is None:
        summary["chat_service_available"] = False
        summary["chat_service_error"] = _chat_service_error
    else:
        summary["chat_service_available"] = True
    return summary


# 静态文件（必须在最后注册，捕获所有未匹配的路由）
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
