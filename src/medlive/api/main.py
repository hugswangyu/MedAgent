"""前端管理 API 启动入口。"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """启动前端管理 API。"""

    if any(arg in {"--help", "-h"} for arg in os.sys.argv[1:]):
        print("用法: uv run medlive-api")
        print("环境变量: LIVERAG_API_HOST=127.0.0.1 LIVERAG_API_PORT=9821")
        return
    uvicorn.run(
        "medlive.api.server:app",
        host=os.getenv("LIVERAG_API_HOST", "127.0.0.1"),
        port=int(os.getenv("LIVERAG_API_PORT", "9821")),
        reload=False,
    )
