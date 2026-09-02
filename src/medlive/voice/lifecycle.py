"""Lease heartbeat and idempotent release for claimed Voice Sessions."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

from medlive.control_plane import ControlPlaneClient, ControlPlaneError

logger = logging.getLogger("agent")


def _heartbeat_interval_seconds() -> float:
    try:
        return max(1.0, float(os.getenv("LIVERAG_WORKER_HEARTBEAT_SECONDS", "30")))
    except ValueError:
        return 30.0


class WorkerSessionLifecycle:
    """Keep a claimed binding leased and release it exactly once."""

    def __init__(
        self,
        control_plane: ControlPlaneClient,
        worker_token: str,
        *,
        heartbeat_interval_s: float | None = None,
    ) -> None:
        self.control_plane = control_plane
        self._worker_token = worker_token
        self.heartbeat_interval_s = (
            heartbeat_interval_s
            if heartbeat_interval_s is not None
            else _heartbeat_interval_seconds()
        )
        self._refresh_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if not self._closed and self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat(), name="voice-session-lease-heartbeat"
            )

    async def current_token(self) -> str:
        async with self._refresh_lock:
            return self._worker_token

    async def refresh_now(self) -> str:
        async with self._refresh_lock:
            if self._closed:
                return self._worker_token
            self._worker_token = await self.control_plane.refresh_worker_token(
                self._worker_token
            )
            return self._worker_token

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        task = self._heartbeat_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        async with self._refresh_lock:
            token = self._worker_token
            try:
                token = await self.control_plane.refresh_worker_token(token)
                self._worker_token = token
            except ControlPlaneError as exc:
                logger.warning(
                    "voice_session.final_refresh_failed", extra={"error": str(exc)}
                )
            try:
                await self.control_plane.end_worker_session(token)
            except ControlPlaneError as exc:
                logger.warning(
                    "voice_session.end_failed", extra={"error": str(exc)}
                )

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            try:
                await self.refresh_now()
            except ControlPlaneError as exc:
                logger.warning(
                    "voice_session.heartbeat_failed", extra={"error": str(exc)}
                )
