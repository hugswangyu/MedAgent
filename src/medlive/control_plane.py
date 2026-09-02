"""HTTP client for the MedAgent PostgreSQL identity/ownership control plane."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import aiohttp


class ControlPlaneError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ControlPlaneClient:
    def __init__(self, base_url: str | None = None, timeout_s: float = 5.0) -> None:
        self.base_url = (
            base_url
            or os.getenv("MEDAGENT_CONTROL_BASE_URL")
            or os.getenv("MEDAGENT_INTERNAL_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.timeout_s = timeout_s
        self.service_key = os.getenv("MEDAGENT_CONTROL_PLANE_KEY", "").strip()

    async def me(self, access_token: str) -> dict[str, Any]:
        return await self._request("GET", "/control/v1/me", token=access_token)

    async def register_knowledge_base(self, access_token: str, kb_id: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/control/v1/knowledge-bases", token=access_token, json={"kb_id": kb_id}
        )

    async def list_knowledge_bases(self, access_token: str) -> dict[str, Any]:
        return await self._request("GET", "/control/v1/knowledge-bases", token=access_token)

    async def get_knowledge_base(self, access_token: str, kb_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/control/v1/knowledge-bases/{quote(kb_id, safe='')}", token=access_token
        )

    async def set_knowledge_base_status(
        self, access_token: str, kb_id: str, status: str
    ) -> dict[str, Any]:
        return await self._request(
            "PUT",
            f"/control/v1/knowledge-bases/{quote(kb_id, safe='')}/status",
            token=access_token,
            json={"status": status},
        )

    async def create_voice_session(
        self, access_token: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/control/v1/voice-sessions", token=access_token, json=payload
        )

    async def get_voice_session(self, access_token: str, session_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/control/v1/voice-sessions/{quote(session_id, safe='')}", token=access_token
        )

    async def end_voice_session(self, access_token: str, session_id: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/control/v1/voice-sessions/{quote(session_id, safe='')}/end",
            token=access_token,
        )

    async def claim_voice_session(
        self, bootstrap_token: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/control/v1/worker/voice-sessions/claim",
            token=bootstrap_token,
            json=payload,
        )

    async def refresh_worker_token(self, worker_token: str) -> str:
        result = await self._request(
            "POST", "/control/v1/worker/token", token=worker_token
        )
        return str(result["worker_token"])

    async def end_worker_session(self, worker_token: str) -> None:
        await self._request(
            "POST", "/control/v1/worker/voice-sessions/end", token=worker_token
        )

    async def cleanup_stale_voice_sessions(self) -> int:
        result = await self._request(
            "POST", "/control/v1/internal/voice-sessions/cleanup", token=None
        )
        return int(result.get("expired", 0))

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str | None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if self.service_key:
            headers["X-Control-Plane-Key"] = self.service_key
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.request(
                    method, f"{self.base_url}{path}", headers=headers, json=json
                ) as response,
            ):
                try:
                    payload = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    payload = {}
                if response.status >= 400:
                    detail = payload.get("detail") if isinstance(payload, dict) else None
                    raise ControlPlaneError(response.status, str(detail or "control plane request failed"))
        except ControlPlaneError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise ControlPlaneError(503, "MedAgent control plane unavailable") from exc
        if not isinstance(payload, dict):
            raise ControlPlaneError(502, "invalid control plane response")
        return payload
