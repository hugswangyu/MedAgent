"""RAG 服务命令行入口。"""

from __future__ import annotations

import os
import sys

import uvicorn

from medlive.rag.settings import Settings


def main() -> None:
    """启动 LightRAG Core Service。"""

    if any(arg in {"--help", "-h"} for arg in sys.argv[1:]):
        print("用法: uv run medlive-rag-service")
        print("环境变量: KB_SERVICE_HOST=127.0.0.1 KB_SERVICE_PORT=9721")
        return
    settings = Settings()
    uvicorn.run(
        "medlive.rag.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=os.getenv("RAG_UVICORN_LOG_LEVEL", "info"),
    )
