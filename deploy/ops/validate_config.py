"""Reject development credentials when the deployment declares production mode."""

from __future__ import annotations

import os
from pathlib import Path


EXAMPLE_SECRETS = {
    "JWT_SECRET_KEY": "phase5-local-jwt-secret-replace-before-shared-use-0001",
    "PG_PASSWORD": "phase5-local-postgres-password",
    "MEDAGENT_INTERNAL_API_KEY": "phase5-local-internal-api-key-replace-0001",
    "MEDAGENT_CONTROL_PLANE_KEY": "phase5-local-control-plane-key-replace-0001",
    "LIVEKIT_API_KEY": "phase5devkey",
    "LIVEKIT_API_SECRET": "phase5-local-livekit-secret-replace-before-shared-use",
    "LIGHTRAG_API_KEY": "phase5-local-lightrag-api-key-replace-0001",
    "ELASTIC_PASSWORD": "phase5-local-elasticsearch-password",
}
DEFAULT_DATA_CREDENTIALS = {
    "NEO4J_PASSWORD": "phase5-local-neo4j-only",
    "MINIO_ROOT_USER": "phase5-local-minio",
    "MINIO_ROOT_PASSWORD": "phase5-local-minio-password",
}

SECRET_FILES = {
    "JWT_SECRET_KEY": "jwt_secret",
    "PG_PASSWORD": "postgres_password",
    "MEDAGENT_INTERNAL_API_KEY": "internal_api_key",
    "MEDAGENT_CONTROL_PLANE_KEY": "control_plane_key",
    "LIVEKIT_API_KEY": "livekit_api_key",
    "LIVEKIT_API_SECRET": "livekit_api_secret",
    "LIGHTRAG_API_KEY": "lightrag_api_key",
    "ELASTIC_PASSWORD": "elasticsearch_password",
}


def _deployment_environment(
    environment: dict[str, str] | None = None,
    secrets_dir: Path = Path("/run/secrets"),
) -> dict[str, str]:
    env = dict(os.environ if environment is None else environment)
    for name, filename in SECRET_FILES.items():
        if env.get(name, "").strip():
            continue
        path = secrets_dir / filename
        try:
            env[name] = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            pass
    return env


def production_errors(
    environment: dict[str, str] | None = None,
    secrets_dir: Path = Path("/run/secrets"),
) -> list[str]:
    env = _deployment_environment(environment, secrets_dir)
    if env.get("MEDRAG_ENV", "dev").strip().lower() != "prod":
        return []
    errors: list[str] = []
    for name, example in EXAMPLE_SECRETS.items():
        value = env.get(name, "").strip()
        if not value:
            errors.append(f"{name} is missing")
        elif value == example or value.startswith("phase5-local-"):
            errors.append(f"{name} uses a repository example value")
    for name, default in DEFAULT_DATA_CREDENTIALS.items():
        value = env.get(name, "").strip()
        if not value:
            errors.append(f"{name} is missing")
        elif value == default or value.startswith("phase5-local-"):
            errors.append(f"{name} uses the default development credential")
    return errors


def main() -> int:
    errors = production_errors()
    if errors:
        raise SystemExit("production configuration rejected: " + "; ".join(errors))
    print("deployment configuration accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
