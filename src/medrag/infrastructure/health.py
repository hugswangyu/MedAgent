"""轻量组件健康追踪器。

用法::

    from medrag.infrastructure.health import report_ok, report_down, get_summary

    report_ok("milvus")
    report_down("embedding", "OOM on model load")
    summary = get_summary()
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

_status: Dict[str, str] = {}
_errors: Dict[str, str] = {}


def register(component: str, initial_status: str = "ok") -> None:
    if component not in _status:
        _status[component] = initial_status


def report_ok(component: str) -> None:
    _status[component] = "ok"
    _errors.pop(component, None)


def report_degraded(component: str, error: str = "") -> None:
    _status[component] = "degraded"
    if error:
        _errors[component] = error


def report_down(component: str, error: str = "") -> None:
    _status[component] = "down"
    if error:
        _errors[component] = error


def get_summary() -> Dict:
    overall = "ok"
    for s in _status.values():
        if s == "down":
            overall = "down"
            break
        if s == "degraded":
            overall = "degraded"

    return {
        "status": overall,
        "components": {
            name: {
                "status": _status.get(name, "unknown"),
                "error": _errors.get(name, ""),
            }
            for name in sorted(_status)
        },
    }
