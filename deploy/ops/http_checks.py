"""Dependency-free Phase 5 prewarm and smoke checks."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def request(url: str, api_key: str = "", attempts: int = 20) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=5
            ) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"check failed for {url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "prewarm"))
    parser.add_argument("--medagent", default=os.getenv("MEDAGENT_URL", "http://localhost:8000"))
    parser.add_argument("--medlive", default=os.getenv("MEDLIVE_URL", "http://localhost:9821"))
    parser.add_argument("--lightrag", default=os.getenv("LIGHTRAG_URL", "http://localhost:9721"))
    parser.add_argument("--kb-id", default="")
    parser.add_argument("--allow-provider-missing", action="store_true")
    args = parser.parse_args()

    request(f"{args.medagent.rstrip('/')}/health")
    request(f"{args.medlive.rstrip('/')}/health")
    key = os.getenv("LIGHTRAG_API_KEY", "")
    request(f"{args.lightrag.rstrip('/')}/v1/healthz", key)
    ready = request(f"{args.lightrag.rstrip('/')}/v1/readyz", key)
    is_ready = bool((ready.get("data") or {}).get("ready"))
    if args.mode == "prewarm" and args.kb_id:
        request(
            f"{args.lightrag.rstrip('/')}/v1/knowledge-bases/{args.kb_id}/ready",
            key,
        )
    if not is_ready and not args.allow_provider_missing:
        raise RuntimeError("LightRAG is live but not ready; configure LLM/embedding providers")
    print(f"phase5 {args.mode} passed (lightrag_ready={is_ready})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
