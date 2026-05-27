"""认证管理器：bcrypt 密码哈希 + JWT 签发/验证。

复用 ``medrag.auth.credentials`` 中的 JSON 文件存储，但将所有密码
升级为 bcrypt 哈希。首次启动时自动迁移已有的明文密码。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import bcrypt
from jose import JWTError, jwt

from medrag.auth.credentials import (
    Credentials,
    load_credentials,
    save_credentials,
)

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "medrag-dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 天

_STORAGE_FILE = os.path.join("tmp_data", "user_credentials.json")


@dataclass
class AuthUser:
    """轻量级认证用户视图（不暴露密码）。"""
    username: str
    is_admin: bool = False


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

def create_access_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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
    # 如果没有任何用户，初始化 admin/admin123
    if not creds:
        admin = Credentials(
            username="admin",
            password=hash_password("admin123"),
            is_admin=True,
        )
        creds["admin"] = admin
        save_credentials(creds, _STORAGE_FILE)
        logger.info("已创建默认管理员账号 admin/admin123")
