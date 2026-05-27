"""FastAPI 应用入口。

启动::

    uvicorn medrag.app.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import auth, chat, sessions, documents
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

    yield

    logger.info("服务器关闭")


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"

app = FastAPI(
    title="Medical RAG Q&A System",
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


@app.get("/health")
async def health():
    return {
        "status": "ok" if _chat_service else "degraded",
        "chat_service_error": _chat_service_error,
    }


# 静态文件（必须在最后注册，捕获所有未匹配的路由）
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
