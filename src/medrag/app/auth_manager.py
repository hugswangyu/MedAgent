"""认证管理器：PostgreSQL 用户、访问 JWT 与短期 worker JWT。"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from medrag.auth.credentials import (
    load_credentials,
)
from medrag.config.settings import settings
from medrag.infrastructure.storage import phase1_repository

logger = logging.getLogger(__name__)

DEFAULT_DEV_SECRET_KEY = "medrag-dev-secret-change-me"
SUPPORTED_ENVIRONMENTS = frozenset({"dev", "test", "prod"})
MIN_PRODUCTION_SECRET_LENGTH = 32
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 天
WORKER_TOKEN_EXPIRE_SECONDS = 300
WORKER_AUDIENCE = "medagent-internal"
WORKER_SCOPES = frozenset(
    {
        "safety:input",
        "safety:output",
        "medical:retrieve",
        "medical:tools",
    }
)

_STORAGE_FILE = str(settings.credentials_path)


@dataclass
class AuthUser:
    """轻量级认证用户视图（不暴露密码）。"""
    user_id: str
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
    user: AuthUser | str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    config = load_auth_config()
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    if isinstance(user, AuthUser):
        user_id = user.user_id
        username = user.username
    else:
        user_id = str(user)
        username = ""
    payload = {
        "sub": user_id,
        "username": username,
        "token_use": "access",
        "iat": now,
        "exp": expire,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, config.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    config = load_auth_config()
    try:
        payload = jwt.decode(token, config.secret_key, algorithms=[ALGORITHM])
        if payload.get("token_use") not in {None, "access"}:
            return None
        return payload
    except JWTError:
        return None


def create_worker_token(
    *,
    user_id: str,
    session_id: str,
    knowledge_base_id: str,
    scopes: set[str] | frozenset[str] = WORKER_SCOPES,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """签发仅供单个 Voice Session 使用的分钟级 worker token。"""

    requested_scopes = set(scopes)
    if not requested_scopes or not requested_scopes.issubset(WORKER_SCOPES):
        raise ValueError("worker token scopes are invalid")
    config = load_auth_config()
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta or timedelta(seconds=WORKER_TOKEN_EXPIRE_SECONDS)
    )
    payload = {
        "sub": user_id,
        "sid": session_id,
        "kid": knowledge_base_id,
        "scope": " ".join(sorted(requested_scopes)),
        "token_use": "worker",
        "aud": WORKER_AUDIENCE,
        "iat": now,
        "exp": expire,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, config.secret_key, algorithm=ALGORITHM)


def decode_worker_token(token: str) -> Optional[dict]:
    config = load_auth_config()
    try:
        payload = jwt.decode(
            token,
            config.secret_key,
            algorithms=[ALGORITHM],
            audience=WORKER_AUDIENCE,
        )
    except JWTError:
        return None
    if payload.get("token_use") != "worker":
        return None
    required = {"sub", "sid", "kid", "scope", "jti", "exp"}
    if not required.issubset(payload):
        return None
    return payload


# ---------------------------------------------------------------------------
# 用户管理
# ---------------------------------------------------------------------------

def get_user(username: str) -> Optional[AuthUser]:
    user = phase1_repository.get_user_by_username(username)
    if user is None:
        return None
    return AuthUser(
        user_id=user.user_id,
        username=user.username,
        is_admin=user.is_admin,
    )


def get_user_by_id(user_id: str) -> Optional[AuthUser]:
    user = phase1_repository.get_user_by_id(user_id)
    if user is None:
        return None
    return AuthUser(
        user_id=user.user_id,
        username=user.username,
        is_admin=user.is_admin,
    )


def verify_user(username: str, plain_password: str) -> Optional[AuthUser]:
    user = phase1_repository.get_user_by_username(username)
    if user is None:
        return None
    if verify_password(plain_password, user.password_hash):
        return AuthUser(
            user_id=user.user_id,
            username=user.username,
            is_admin=user.is_admin,
        )
    return None


def create_user(username: str, password: str, is_admin: bool = False) -> Optional[AuthUser]:
    user = phase1_repository.create_user(username, hash_password(password), is_admin)
    if user is None:
        return None
    return AuthUser(
        user_id=user.user_id,
        username=user.username,
        is_admin=user.is_admin,
    )


# ---------------------------------------------------------------------------
# 启动迁移：将明文密码升级为 bcrypt
# ---------------------------------------------------------------------------

def init_auth() -> None:
    """安装 schema 并把旧 JSON 用户复制到 PostgreSQL；保留原文件。"""

    load_auth_config()
    creds = load_credentials(_STORAGE_FILE)
    phase1_repository.ensure_schema()
    legacy_rows = []
    for user in creds.values():
        password_hash = (
            user.password if user.password.startswith("$2") else hash_password(user.password)
        )
        legacy_rows.append((user.username, password_hash, user.is_admin))
    imported = phase1_repository.import_legacy_users(legacy_rows)
    logger.info(
        "PostgreSQL 身份初始化完成：legacy_users=%s imported=%s legacy_file=preserved",
        len(legacy_rows),
        imported,
    )
