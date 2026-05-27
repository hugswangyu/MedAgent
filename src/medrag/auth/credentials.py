"""User credentials model and JSON file persistence.

Replaces the scattered auth logic previously in ``user_data_storage.py``
and the hardcoded admin check in ``webui.py``.
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
# Persistence
# ---------------------------------------------------------------------------

DEFAULT_STORAGE_FILE = os.path.join("tmp_data", "user_credentials.json")


def _ensure_storage_dir(file_path: str) -> None:
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def load_credentials(file_path: str = DEFAULT_STORAGE_FILE) -> Dict[str, Credentials]:
    """Read credentials from *file_path*.  Returns an empty dict on failure."""
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
    """Write *credentials* dict to *file_path*."""
    _ensure_storage_dir(file_path)
    data = {k: v.to_dict() for k, v in credentials.items()}
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_or_create_credentials(
    file_path: str = DEFAULT_STORAGE_FILE,
) -> Dict[str, Credentials]:
    """Load credentials, initialising a default admin account if the file is empty."""
    creds = load_credentials(file_path)
    if not creds:
        admin = Credentials(username="admin", password="admin123", is_admin=True)
        creds["admin"] = admin
        save_credentials(creds, file_path)
    return creds
