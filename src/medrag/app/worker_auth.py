"""阶段 1 worker token 校验、会话绑定与 PostgreSQL 防重放。"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from medrag.app.auth_manager import decode_worker_token
from medrag.infrastructure.storage import phase1_repository


@dataclass(frozen=True)
class WorkerPrincipal:
    """已经过签名、scope、会话和 nonce 校验的 worker 身份。"""

    user_id: str
    session_id: str
    knowledge_base_id: str
    token_jti: str


class WorkerAuthorizationError(ValueError):
    """worker 请求拒绝；message 可安全返回，不含密钥或 token。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def authorize_worker_request(
    *,
    authorization: str | None,
    nonce: str | None,
    required_scope: str,
    payload: dict[str, Any],
) -> WorkerPrincipal:
    """验证短期 token 并原子消费请求 nonce。"""

    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise WorkerAuthorizationError("UNAUTHORIZED", "缺少 worker token")
    claims = decode_worker_token(authorization[len(prefix) :].strip())
    if claims is None:
        raise WorkerAuthorizationError("UNAUTHORIZED", "worker token 无效或已过期")
    scopes = set(str(claims["scope"]).split())
    if required_scope not in scopes:
        raise WorkerAuthorizationError("FORBIDDEN", "worker token scope 不允许该操作")
    session_id = str(payload.get("session_id") or "")
    if session_id != str(claims["sid"]):
        raise WorkerAuthorizationError("SESSION_BINDING_MISMATCH", "Voice Session 绑定不匹配")
    try:
        user_id = str(uuid.UUID(str(claims["sub"])))
        token_jti = str(uuid.UUID(str(claims["jti"])))
        nonce_value = str(uuid.UUID(str(nonce or "")))
    except ValueError as exc:
        raise WorkerAuthorizationError("UNAUTHORIZED", "worker token 或 nonce 格式无效") from exc
    knowledge_base_id = str(claims["kid"])
    expires_at = datetime.fromtimestamp(float(claims["exp"]), tz=timezone.utc)
    accepted = phase1_repository.consume_worker_nonce(
        token_jti=token_jti,
        nonce=nonce_value,
        user_id=user_id,
        session_id=session_id,
        request_hash=phase1_repository.request_digest(payload),
        expires_at=expires_at,
    )
    if not accepted:
        raise WorkerAuthorizationError("REPLAY_DETECTED", "重复请求 nonce 已被拒绝")
    phase1_repository.bind_voice_session(
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=knowledge_base_id,
    )
    return WorkerPrincipal(
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=knowledge_base_id,
        token_jti=token_jti,
    )


def allow_legacy_internal_key() -> bool:
    """阶段 1 迁移开关；生产默认关闭共享 key，开发默认兼容。"""

    configured = os.getenv("MEDAGENT_ALLOW_LEGACY_INTERNAL_API_KEY")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return os.getenv("MEDRAG_ENV", "dev").strip().lower() != "prod"
