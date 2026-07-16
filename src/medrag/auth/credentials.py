"""用户凭据模型与 JSON 文件持久化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from medrag.config.settings import settings
from medrag.infrastructure.storage import JsonStore


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

_default_store = JsonStore(str(settings.credentials_path))


def load_credentials(file_path: str | None = None) -> Dict[str, Credentials]:
    store = JsonStore(file_path) if file_path else _default_store
    data = store.read()
    if isinstance(data, list):
        return {}
    return {k: Credentials.from_dict(v) for k, v in data.items()}


def save_credentials(
    credentials: Dict[str, Credentials],
    file_path: str | None = None,
) -> None:
    store = JsonStore(file_path) if file_path else _default_store
    data = {k: v.to_dict() for k, v in credentials.items()}
    store.write(data)


def get_or_create_credentials(
    file_path: str | None = None,
) -> Dict[str, Credentials]:
    """Compatibility alias that no longer creates default credentials."""
    return load_credentials(file_path)
