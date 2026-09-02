"""随主进程启动内置 RAG HTTP 服务。"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from threading import Lock, Thread
from typing import Any

import uvicorn

from medlive.rag.settings import Settings as RagSettings

logger = logging.getLogger("medlive.rag.service")
_START_LOCK = Lock()
_START_THREAD: Thread | None = None


class RagServiceStartStatus(str, Enum):
    """内置 RAG 服务启动状态。"""

    DISABLED = "disabled"
    ALREADY_RUNNING = "already_running"
    STARTING = "starting"
    STARTED = "started"


@dataclass(frozen=True)
class RagReadyState:
    """描述一次 RAG ready 检查结果。"""

    ready: bool
    status: str
    data: dict[str, Any] | None = None
    error: str | None = None


def port_is_open(host: str, port: int) -> bool:
    """检查端口是否可连接。"""

    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def start_embedded_rag_service() -> RagServiceStartStatus:
    """启动内置 RAG 服务，已有服务时直接复用。"""

    if os.getenv("LIGHTRAG_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return RagServiceStartStatus.DISABLED
    settings = RagSettings()
    if port_is_open(settings.host, settings.port):
        logger.info("rag.service.already_running", extra={"host": settings.host, "port": settings.port})
        return RagServiceStartStatus.ALREADY_RUNNING

    with _START_LOCK:
        global _START_THREAD
        if _START_THREAD is not None and _START_THREAD.is_alive():
            return RagServiceStartStatus.STARTING
        if port_is_open(settings.host, settings.port):
            logger.info("rag.service.already_running", extra={"host": settings.host, "port": settings.port})
            return RagServiceStartStatus.ALREADY_RUNNING

        def _run_server() -> None:
            uvicorn.run(
                "medlive.rag.server:app",
                host=settings.host,
                port=settings.port,
                reload=False,
                log_level=os.getenv("RAG_UVICORN_LOG_LEVEL", "warning"),
            )

        _START_THREAD = Thread(target=_run_server, name="rag-service", daemon=True)
        _START_THREAD.start()
        logger.info(
            "rag.service.starting",
            extra={"host": settings.host, "port": settings.port, "working_dir": settings.absolute_working_dir},
        )
        return RagServiceStartStatus.STARTED


def wait_for_rag_ready(*, timeout_ms: int = 15000, interval_ms: int = 250) -> RagReadyState:
    """等待内部 RAG 服务进入 ready 状态。"""

    start_status = start_embedded_rag_service()
    if start_status == RagServiceStartStatus.DISABLED:
        return RagReadyState(ready=False, status=start_status.value, error="RAG 服务已禁用")

    settings = RagSettings()
    deadline = time.monotonic() + max(timeout_ms, 1) / 1000.0
    last_error = ""
    headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
    url = f"http://{settings.host}:{settings.port}/v1/readyz"

    while time.monotonic() <= deadline:
        try:
            request = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(request, timeout=max(interval_ms / 1000.0, 0.1)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict) and data.get("ready") is True:
                return RagReadyState(ready=True, status=start_status.value, data=data)
            last_error = "RAG 服务尚未 ready"
        except urllib.error.HTTPError as exc:
            last_error = f"readyz 返回 HTTP {exc.code}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(interval_ms / 1000.0)

    return RagReadyState(ready=False, status=start_status.value, error=last_error or "等待 RAG ready 超时")
