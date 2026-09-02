"""访问 JWT 校验与 Voice worker 短期 JWT 签发。"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

ALGORITHM = "HS256"
WORKER_AUDIENCE = "medagent-internal"
WORKER_TOKEN_TTL_SECONDS = 300
WORKER_SCOPES = (
    "medical:retrieve medical:tools safety:input safety:output"
)
_bearer = HTTPBearer()
BEARER_CREDENTIALS = Depends(_bearer)


@dataclass(frozen=True)
class CurrentUser:
    """由 MedAgent 访问 JWT 得到的不可变用户身份。"""

    user_id: str
    username: str
    is_admin: bool = False
    token_version: int = 1
    access_token: str = ""


def _secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY 必须与 MedAgent 显式配置为同一强密钥")
    return secret


def decode_access_token(token: str) -> CurrentUser:
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        if payload.get("token_use") not in {None, "access"}:
            raise JWTError("invalid token use")
        user_id = str(uuid.UUID(str(payload.get("sub") or "")))
    except (JWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期或身份凭据无效",
        ) from exc
    return CurrentUser(
        user_id=user_id,
        username=str(payload.get("username") or ""),
        is_admin=bool(payload.get("is_admin", False)),
        token_version=int(payload.get("ver", 0)),
        access_token=token,
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = BEARER_CREDENTIALS,
) -> CurrentUser:
    local_user = decode_access_token(credentials.credentials)
    from medlive.control_plane import ControlPlaneClient, ControlPlaneError

    try:
        authoritative = await ControlPlaneClient().me(credentials.credentials)
    except ControlPlaneError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return CurrentUser(
        user_id=str(authoritative["user_id"]),
        username=str(authoritative.get("username") or local_user.username),
        is_admin=bool(authoritative.get("is_admin", False)),
        token_version=int(authoritative.get("token_version", 1)),
        access_token=credentials.credentials,
    )


CURRENT_USER = Depends(get_current_user)


async def get_current_admin(
    current_user: CurrentUser = CURRENT_USER,
) -> CurrentUser:
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return current_user
