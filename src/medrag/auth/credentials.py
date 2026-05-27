"""用户凭据模型与 JSON 文件持久化。

替代分散在 ``user_data_storage.py`` 中的认证逻辑和 ``webui.py`` 中硬编码的管理员检查。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class Credentials:
    username: str
    password: str
    is_admin: bool = False

    def to_dict(self) -> Dict:
        return {
            "username": self.username,
            "password": self.password,
            "is_admin": self.is_admin,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Credentials":
        return cls(
            username=data["username"],
            password=data["password"],
            is_admin=data.get("is_admin", False),
        )


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------

DEFAULT_STORAGE_FILE = os.path.join("tmp_data", "user_credentials.json")


def _ensure_storage_dir(file_path: str) -> None:
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def load_credentials(file_path: str = DEFAULT_STORAGE_FILE) -> Dict[str, Credentials]:
    """从 *file_path* 读取凭据，失败时返回空字典。"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return {k: Credentials.from_dict(v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_credentials(
    credentials: Dict[str, Credentials],
    file_path: str = DEFAULT_STORAGE_FILE,
) -> None:
    """将 *credentials* 字典写入 *file_path*。"""
    _ensure_storage_dir(file_path)
    data = {k: v.to_dict() for k, v in credentials.items()}
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_or_create_credentials(
    file_path: str = DEFAULT_STORAGE_FILE,
) -> Dict[str, Credentials]:
    """加载凭据，若文件为空则初始化默认管理员账号。"""
    creds = load_credentials(file_path)
    if not creds:
        admin = Credentials(username="admin", password="admin123", is_admin=True)
        creds["admin"] = admin
        save_credentials(creds, file_path)
    return creds
