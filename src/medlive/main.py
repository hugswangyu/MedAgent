"""LiveRAG 统一启动入口。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import aiohttp
from livekit.agents import AgentServer, JobContext, cli, room_io

from medlive.agent.assistant import VoiceAssistant
from medlive.agent.metrics_hooks import (
    MetricsState,
    register_session_metrics_hooks,
    start_network_probe_task,
)
from medlive.agent.providers import build_agent_session
from medlive.agent.tool import MedicalCapabilityClient, RagClient
from medlive.config.settings import load_app_settings, load_environment, public_voice_config
from medlive.context.manager import ContextManager
from medlive.context.renderer import SessionPromptRenderer
from medlive.context.store import ContextStore
from medlive.control_plane import ControlPlaneClient, ControlPlaneError
from medlive.logging.events import EventLogger
from medlive.logging.setup import setup_logging
from medlive.rag.service import wait_for_rag_ready
from medlive.runtime.paths import build_runtime_paths
from medlive.voice.lifecycle import WorkerSessionLifecycle

load_environment()
setup_logging()
logger = logging.getLogger("agent")
server = AgentServer()


class _WorkerShutdownCoordinator:
    """Ensure pending voice audits are flushed before releasing the binding."""

    def __init__(self, lifecycle: WorkerSessionLifecycle) -> None:
        self._lifecycle = lifecycle
        self._audit_flush: Callable[[], Awaitable[None]] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()

    def attach_audit_flush(
        self, audit_flush: Callable[[], Awaitable[None]]
    ) -> None:
        """Attach the assistant flush hook before the voice session starts."""

        if self._close_task is not None:
            raise RuntimeError("worker shutdown already started")
        self._audit_flush = audit_flush

    async def current_token(self) -> str:
        return await self._lifecycle.current_token()

    def start(self) -> None:
        self._lifecycle.start()

    async def close(self) -> None:
        """Run one shared shutdown, independent of callback execution order."""

        async with self._close_lock:
            if self._close_task is None:
                self._close_task = asyncio.create_task(
                    self._flush_then_release(),
                    name="voice-audit-flush-and-binding-release",
                )
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _flush_then_release(self) -> None:
        try:
            if self._audit_flush is not None:
                await self._audit_flush()
        finally:
            await self._lifecycle.close()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext) -> None:
    """my-agent 在线语音会话入口。"""

    settings = load_app_settings()
    paths = build_runtime_paths(settings.user_data_dir)
    store = ContextStore(paths)
    store.initialize(reset_session=False)
    await asyncio.to_thread(wait_for_rag_ready, timeout_ms=settings.api.rag_ready_timeout_ms)
    control_plane = ControlPlaneClient()
    voice_session = await _voice_session_from_metadata(ctx, control_plane)
    lifecycle = await _start_worker_lifecycle(ctx, control_plane, voice_session)
    if voice_session is not None:
        store = store.for_voice_session(str(voice_session["session_id"]))
    else:
        store = store.for_ephemeral_session()
    store.clear_session()
    knowledge_base = await _resolve_knowledge_base(settings, voice_session)

    session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_log_path = paths.logs_dir / f"{session_stamp}_metrics_{ctx.job.id}_{ctx.job.room.sid}.jsonl"
    event_logger = EventLogger(
        metrics_log_path,
        room=ctx.room.name if ctx.room else None,
        room_id=ctx.job.room.sid if ctx.job and ctx.job.room else None,
        job_id=ctx.job.id if ctx.job else None,
        agent_name="my-agent",
    )

    session = build_agent_session(settings)
    rag_client = RagClient(
        settings.rag,
        store,
        user_data_dir=settings.user_data_dir,
        kb_id=knowledge_base["kb_id"],
        kb_name=knowledge_base["name"],
    )
    prompt_result = SessionPromptRenderer(store=store, history_limit=settings.history_limit).render(
        kb_id=knowledge_base["kb_id"],
        kb_name=knowledge_base["name"],
        rag_tool_mode=settings.rag.rag_tool_mode,
    )
    overview_state = {
        "generated": False,
        "fallback": False,
        "reason": "startup_read_only",
        "meta": store.read_knowledge_overview_meta(knowledge_base["kb_id"]),
    }

    active_voice = public_voice_config(settings.voice, effective="active_session")
    state = store.read_runtime_state()
    state.update(
        {
            "active_session": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "session_id": voice_session.get("session_id") if voice_session else None,
                "user_id": voice_session.get("user_id") if voice_session else None,
                "client_type": voice_session.get("client_type") if voice_session else "web",
                "job_id": ctx.job.id if ctx.job else None,
                "room_id": ctx.job.room.sid if ctx.job and ctx.job.room else None,
                "room": ctx.room.name if ctx.room else None,
                "voice": active_voice,
                "knowledge_base": {
                    "kb_id": knowledge_base["kb_id"],
                    "name": knowledge_base["name"],
                    "locked_at": datetime.now(timezone.utc).isoformat(),
                    "job_id": ctx.job.id if ctx.job else None,
                    "room_id": ctx.job.room.sid if ctx.job and ctx.job.room else None,
                },
                "session_prompt_chars": prompt_result.prompt_chars,
                "history_count": prompt_result.history_count,
                "knowledge_overview": overview_state,
            },
            "active_voice_model": active_voice,
            "model_pending_reconnect": False,
            "rag_tool_mode": settings.rag.rag_tool_mode,
        }
    )
    store.write_runtime_state(state)
    event_logger.append("model.active_session", {"voice": active_voice})
    event_logger.append("knowledge_base.active_session", knowledge_base)
    event_logger.append(
        "context.session_prompt.rendered",
        {
            "kb_id": prompt_result.kb_id,
            "kb_name": prompt_result.kb_name,
            "prompt_chars": prompt_result.prompt_chars,
            "history_count": prompt_result.history_count,
            "rag_tool_mode": prompt_result.rag_tool_mode,
            "overview_generated": False,
            "overview_fallback": False,
            "overview_generation_timing": "index_completed_only",
        },
    )

    context_manager = ContextManager(store=store, rag_client=rag_client)
    capability_session_id = (
        str(voice_session.get("session_id"))
        if voice_session is not None
        else f"job_{ctx.job.id}"
    )
    worker_token_factory = None
    if lifecycle is not None:
        worker_token_factory = lifecycle.current_token
    medical_client = MedicalCapabilityClient(
        base_url=os.getenv("MEDAGENT_INTERNAL_BASE_URL", "http://127.0.0.1:8000"),
        api_key=os.getenv(
            "MEDAGENT_INTERNAL_API_KEY", ""
        ),
        worker_token_factory=worker_token_factory,
        session_id=capability_session_id,
        input_timeout_ms=_env_int(
            "MEDAGENT_INPUT_SAFETY_TIMEOUT_MS", 400
        ),
        retrieval_timeout_ms=_env_int(
            "MEDAGENT_MEDICAL_RETRIEVAL_TIMEOUT_MS", 1500
        ),
        tool_timeout_ms=_env_int(
            "MEDAGENT_MEDICAL_TOOL_TIMEOUT_MS", 500
        ),
        output_timeout_ms=_env_int(
            "MEDAGENT_OUTPUT_SAFETY_TIMEOUT_MS", 400
        ),
    )

    async def _confirmed_memory_context(turn_id: str) -> list[str]:
        result = await medical_client.confirmed_memory_context(turn_id=turn_id)
        facts = result.data.get("confirmed_facts")
        if not isinstance(facts, list):
            return []
        return [str(item) for item in facts if isinstance(item, str)]

    assistant = VoiceAssistant(
        context_manager=context_manager,
        session_system_prompt=prompt_result.prompt,
        rag_tool_mode=settings.rag.rag_tool_mode,
        medical_client=medical_client,
        event_logger=event_logger,
        memory_context_provider=(
            _confirmed_memory_context if lifecycle is not None else None
        ),
    )
    memory_finalize: dict[str, Any] = {}

    async def _flush_and_finalize_memory() -> None:
        try:
            await assistant.flush_pending_turn_writes()
            result = await medical_client.finalize_session(summary_version=1)
            memory_finalize["result"] = result.data
            event_logger.append(
                "memory.finalize.completed",
                {"request_id": result.request_id, "result": result.data},
            )
        except Exception as exc:
            memory_finalize["error"] = type(exc).__name__
            logger.warning(
                "memory.finalize.failed",
                extra={"error": type(exc).__name__},
            )

    if lifecycle is not None:
        lifecycle.attach_audit_flush(_flush_and_finalize_memory)
    history_compactor = None
    await ctx.connect()
    metrics_state = MetricsState()
    register_session_metrics_hooks(session, logger, event_logger, metrics_state)
    probe_task = start_network_probe_task(
        livekit_url=settings.voice.livekit_url,
        state=metrics_state,
        logger=logger,
        metrics_logger=event_logger,
    )

    async def _finalize_session(reason: str = "") -> None:
        """LiveKit job 结束后收尾并压缩当前通话 history。"""

        if lifecycle is None:
            await assistant.flush_pending_turn_writes()
        else:
            await lifecycle.close()
        if probe_task is not None:
            probe_task.cancel()
            with suppress(asyncio.CancelledError):
                await probe_task
        history_result = await _close_binding_then_compact_history(
            lifecycle,
            history_compactor,
            knowledge_base,
            replacement_result=memory_finalize.get("result"),
        )
        state = store.read_runtime_state()
        active_session = state.get("active_session")
        if isinstance(active_session, dict):
            active_session["ended_at"] = datetime.now(timezone.utc).isoformat()
            active_session["history_compaction"] = history_result
            state["active_session"] = active_session
            store.write_runtime_state(state)
        event_logger.append("history.compact.finalized", {"result": history_result, "reason": reason})
        if voice_session is None:
            store.clear_session()
        await medical_client.aclose()

    ctx.add_shutdown_callback(_finalize_session)
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(sample_rate=16000),
            text_input=room_io.TextInputOptions(),
            text_output=room_io.TextOutputOptions(),
        ),
    )
    if lifecycle is not None:
        lifecycle.start()


async def _start_worker_lifecycle(
    ctx: JobContext,
    control_plane: ControlPlaneClient,
    voice_session: dict[str, Any] | None,
) -> _WorkerShutdownCoordinator | None:
    if voice_session is None:
        return None
    coordinator = _WorkerShutdownCoordinator(
        WorkerSessionLifecycle(
            control_plane, str(voice_session["worker_token"])
        )
    )

    async def release_claimed_session() -> None:
        await coordinator.close()

    try:
        ctx.add_shutdown_callback(release_claimed_session)
    except BaseException:
        await coordinator.close()
        raise
    return coordinator


async def _close_binding_then_compact_history(
    lifecycle: WorkerSessionLifecycle | None,
    history_compactor: Any,
    knowledge_base: dict[str, str],
    *,
    replacement_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Disable independent history only after PostgreSQL replacement succeeds."""

    if lifecycle is not None:
        await lifecycle.close()
        if replacement_result and replacement_result.get("replacement_verified") is True:
            return {
                "updated": False,
                "reason": "postgres_memory_replacement_verified",
                "summary_id": replacement_result.get("summary_id"),
                "summary_version": replacement_result.get("summary_version"),
            }
        return {
            "updated": False,
            "reason": "postgres_memory_replacement_unavailable",
        }
    del history_compactor, knowledge_base
    return {
        "updated": False,
        "reason": "medlive_independent_history_disabled",
    }


async def _resolve_knowledge_base(
    settings: Any,
    voice_session: dict[str, Any] | None = None,
) -> dict[str, str]:
    """读取并预热本次通话锁定的知识库。"""

    if voice_session is not None:
        kb_id = str(
            voice_session.get("knowledge_base_id")
            or voice_session.get("kb_id")
            or "default"
        ).strip() or "default"
    else:
        kb_id = "default"
    detail = await _fetch_knowledge_base(settings, kb_id)
    if detail is None and voice_session is not None:
        raise RuntimeError("Voice Session 绑定的知识库不存在或不可用")
    if detail is None and kb_id != "default":
        kb_id = "default"
        detail = await _fetch_knowledge_base(settings, kb_id)
    if detail is None:
        detail = {"kb_id": "default", "name": "默认知识库"}
    await _preheat_knowledge_base(settings, str(detail["kb_id"]))
    return {"kb_id": str(detail["kb_id"]), "name": str(detail["name"])}


async def _voice_session_from_metadata(
    ctx: JobContext,
    control_plane: ControlPlaneClient,
) -> dict[str, Any] | None:
    """从 Agent dispatch metadata claim 服务端预建绑定；生产默认 fail-closed。"""

    raw = ""
    try:
        raw = str(getattr(ctx.job, "metadata", "") or "")
    except Exception:
        raw = ""
    if not raw:
        try:
            raw = ctx.room.metadata if ctx.room else ""
        except Exception:
            raw = ""
    if not raw:
        return _legacy_unbound_or_raise("Voice worker metadata 缺失")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _legacy_unbound_or_raise("Voice worker metadata 不是合法 JSON")
    required = {"session_id", "binding_version", "worker_bootstrap_token"}
    if not isinstance(data, dict) or not required.issubset(data):
        return _legacy_unbound_or_raise("Voice worker metadata 缺少绑定字段")
    actual_room = ctx.room.name if ctx.room else ""
    if not actual_room:
        raise RuntimeError("Voice Session 缺少实际 room")
    try:
        return await control_plane.claim_voice_session(
            str(data["worker_bootstrap_token"]),
            {
                "session_id": str(data["session_id"]),
                "binding_version": int(data["binding_version"]),
                "room_name": actual_room,
                "livekit_job_id": str(ctx.job.id),
            },
        )
    except (ControlPlaneError, TypeError, ValueError) as exc:
        raise RuntimeError("Voice Session 服务端绑定 claim 失败") from exc


def _legacy_unbound_or_raise(message: str) -> None:
    environment = os.getenv(
        "LIVERAG_ENV", os.getenv("MEDRAG_ENV", "prod")
    ).strip().lower()
    enabled = os.getenv("LIVERAG_ALLOW_UNBOUND_WORKER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enabled and environment in {"dev", "test"}:
        logger.warning(
            "voice_session.unbound_development_mode", extra={"reason": message}
        )
        return None
    raise RuntimeError(message)


async def _fetch_knowledge_base(settings: Any, kb_id: str) -> dict[str, Any] | None:
    """从内部 RAG 服务读取知识库详情。"""

    data = await _rag_get(settings, f"/v1/knowledge-bases/{quote(kb_id, safe='')}")
    if not isinstance(data, dict) or data.get("status") != "ok":
        return None
    payload = data.get("data")
    return payload if isinstance(payload, dict) else None


async def _preheat_knowledge_base(settings: Any, kb_id: str) -> None:
    """预热本次通话要使用的知识库 engine。"""

    await _rag_get(settings, f"/v1/knowledge-bases/{quote(kb_id, safe='')}/ready")


async def _rag_get(settings: Any, path: str) -> dict[str, Any] | None:
    """对内部 RAG 服务执行一次 GET 请求。"""

    headers = {"X-API-Key": settings.rag.api_key} if settings.rag.api_key else {}
    url = f"{settings.rag.base_url.rstrip('/')}{path}"
    timeout = aiohttp.ClientTimeout(total=max(settings.api.rag_ready_timeout_ms, 1000) / 1000.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url, headers=headers) as response:
            if response.status != 200:
                return None
            payload = await response.json()
    except Exception as exc:
        logger.warning("knowledge_base.resolve_failed", extra={"path": path, "error": str(exc)})
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    """运行 LiveKit Agent CLI。"""

    cli.run_app(server)


def _env_int(name: str, default: int) -> int:
    """读取阶段 0 硬超时，非法值回退冻结默认值。"""

    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


if __name__ == "__main__":
    main()
