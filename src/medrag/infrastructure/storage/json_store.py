"""通用 JSON 文件读写，消除 session / document / credential 重复的 I/O 模式。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_path_locks: dict[str, threading.RLock] = {}
_path_locks_guard = threading.Lock()


def _get_path_lock(path: str) -> threading.RLock:
    key = os.path.normcase(path)
    with _path_locks_guard:
        return _path_locks.setdefault(key, threading.RLock())


class JsonStore:
    """单个 JSON 文件的读写门面。"""

    def __init__(self, file_path: str) -> None:
        self._path = os.path.abspath(os.path.normpath(os.path.expanduser(file_path)))
        self._lock = _get_path_lock(self._path)

    @property
    def path(self) -> str:
        return self._path

    def read(self) -> dict | list:
        with self._lock:
            try:
                return self._read_unlocked()
            except (FileNotFoundError, json.JSONDecodeError):
                return {}

    def write(self, data: dict | list) -> None:
        with self._lock:
            self._write_unlocked(data)

    def update(
        self,
        mutator: Callable[[dict | list], T],
        *,
        default_factory: Callable[[], dict | list] = dict,
    ) -> T:
        with self._lock:
            try:
                data = self._read_unlocked()
            except FileNotFoundError:
                data = default_factory()
            result = mutator(data)
            self._write_unlocked(data)
            return result

    def _read_unlocked(self) -> dict | list:
        with open(self._path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _write_unlocked(self, data: dict | list) -> None:
        folder = os.path.dirname(self._path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        temp_path = ""
        try:
            file_descriptor, temp_path = tempfile.mkstemp(
                dir=folder or ".",
                prefix=f".{os.path.basename(self._path)}.",
                suffix=".tmp",
            )
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self._path)
            temp_path = ""
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
