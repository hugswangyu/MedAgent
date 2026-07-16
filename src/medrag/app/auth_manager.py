"""认证管理器：bcrypt 密码哈希 + JWT 签发/验证。

复用 ``medrag.auth.credentials`` 中的 JSON 文件存储，但将所有密码
升级为 bcrypt 哈希。首次启动时自动迁移已有的明文密码。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bcrypt
from jose import JWTError, jwt

from medrag.auth.credentials import (
    Credentials,
    load_credentials,
    save_credentials,
)
from medrag.config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_DEV_SECRET_KEY = "medrag-dev-secret-change-me"
SUPPORTED_ENVIRONMENTS = frozenset({"dev", "test", "prod"})
MIN_PRODUCTION_SECRET_LENGTH = 32
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 天

_STORAGE_FILE = str(settings.credentials_path)


@dataclass
class AuthUser:
    """轻量级认证用户视图（不暴露密码）。"""
    username: str
    is_admin: bool = False


@dataclass(frozen=True)
class AuthConfig:
    """Validated authentication settings. The secret must never be logged."""

    environment: str
    secret_key: str
    allow_public_registration: bool


def load_auth_config(
    environment: Optional[str] = None,
    secret_key: Optional[str] = None,
    allow_public_registration: Optional[bool | str] = None,
) -> AuthConfig:
    """Load and validate authentication settings.

    dev and test retain a compatible local default. prod requires an explicit
    JWT_SECRET_KEY of at least 32 characters that differs from the default.
    """
    raw_environment = (
        environment if environment is not None else os.getenv("MEDRAG_ENV", "dev")
    )
    normalized_environment = raw_environment.strip().lower()
    if normalized_environment not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise RuntimeError(
            f"Invalid MEDRAG_ENV={raw_environment!r}; supported values: {supported}"
        )

    configured_secret = (
        secret_key if secret_key is not None else os.getenv("JWT_SECRET_KEY", "")
    )
    configured_secret = configured_secret.strip()

    if normalized_environment == "prod":
        if not configured_secret:
            raise RuntimeError("JWT_SECRET_KEY is required when MEDRAG_ENV=prod")
        if configured_secret == DEFAULT_DEV_SECRET_KEY:
            raise RuntimeError("The default development JWT secret is forbidden in prod")
        if len(configured_secret) < MIN_PRODUCTION_SECRET_LENGTH:
            raise RuntimeError(
                f"JWT_SECRET_KEY must contain at least {MIN_PRODUCTION_SECRET_LENGTH} characters in prod"
            )

    registration_value = (
        allow_public_registration
        if allow_public_registration is not None
        else os.getenv("ALLOW_PUBLIC_REGISTRATION")
    )
    if registration_value is None or str(registration_value).strip() == "":
        registration_enabled = normalized_environment != "prod"
    elif isinstance(registration_value, bool):
        registration_enabled = registration_value
    else:
        normalized_registration = registration_value.strip().lower()
        if normalized_registration in {"1", "true", "yes", "on"}:
            registration_enabled = True
        elif normalized_registration in {"0", "false", "no", "off"}:
            registration_enabled = False
        else:
            raise RuntimeError(
                "ALLOW_PUBLIC_REGISTRATION must be a boolean value"
            )

    effective_secret = configured_secret or DEFAULT_DEV_SECRET_KEY
    logger.info(
        "Authentication configuration validated: environment=%s, jwt_secret=%s, public_registration=%s",
        normalized_environment,
        "configured" if configured_secret else "not configured (development default)",
        "enabled" if registration_enabled else "disabled",
    )
    return AuthConfig(
        normalized_environment,
        effective_secret,
        registration_enabled,
    )


def is_public_registration_enabled() -> bool:
    """Return whether the unauthenticated registration endpoint is enabled."""
    return load_auth_config().allow_public_registration


# ---------------------------------------------------------------------------
# 密码
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def create_access_token(
    username: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    config = load_auth_config()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, config.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    config = load_auth_config()
    try:
        return jwt.decode(token, config.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# 用户管理
# ---------------------------------------------------------------------------

def get_user(username: str) -> Optional[AuthUser]:
    creds = load_credentials(_STORAGE_FILE)
    user = creds.get(username)
    if user is None:
        return None
    return AuthUser(username=user.username, is_admin=user.is_admin)


def get_user_with_password(username: str) -> Optional[Credentials]:
    return load_credentials(_STORAGE_FILE).get(username)


def verify_user(username: str, plain_password: str) -> Optional[AuthUser]:
    user = get_user_with_password(username)
    if user is None:
        return None
    if verify_password(plain_password, user.password):
        return AuthUser(username=user.username, is_admin=user.is_admin)
    return None


def create_user(username: str, password: str, is_admin: bool = False) -> Optional[AuthUser]:
    creds = load_credentials(_STORAGE_FILE)
    if username in creds:
        return None
    creds[username] = Credentials(
        username=username,
        password=hash_password(password),
        is_admin=is_admin,
    )
    save_credentials(creds, _STORAGE_FILE)
    return AuthUser(username=username, is_admin=is_admin)


# ---------------------------------------------------------------------------
# 启动迁移：将明文密码升级为 bcrypt
# ---------------------------------------------------------------------------

def init_auth() -> None:
    """加载用户数据，迁移明文密码 → bcrypt。"""
    # Validate before touching user data.
    load_auth_config()
    creds = load_credentials(_STORAGE_FILE)
    changed = False
    for name, user in list(creds.items()):
        if user.password.startswith("$2"):
            continue  # 已是 bcrypt
        logger.info("迁移用户 %s 的密码为 bcrypt 哈希", name)
        creds[name] = Credentials(
            username=user.username,
            password=hash_password(user.password),
            is_admin=user.is_admin,
        )
        changed = True
    if changed:
        save_credentials(creds, _STORAGE_FILE)
        logger.info("密码迁移完成")
