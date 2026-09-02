"""标准日志初始化。"""

from __future__ import annotations

import logging


def setup_logging() -> None:
    """初始化基础日志格式。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
