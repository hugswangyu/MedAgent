"""Apply or transactionally dry-run the ordered MedAgent PostgreSQL migrations."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import psycopg2

MIGRATIONS = (
    "create_tables.sql",
    "phase1_identity_data.sql",
    "phase2_safe_voice.sql",
    "phase3_memory.sql",
)
LOCK_ID = 5_005_005


def _connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "postgres"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "medagent"),
        password=os.environ["PG_PASSWORD"],
        dbname=os.getenv("PG_DATABASE", "medagent"),
    )


def run(mode: str, scripts_dir: Path) -> list[str]:
    applied: list[str] = []
    conn = _connection()
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("SELECT name, sha256 FROM schema_migrations")
            known = dict(cur.fetchall())
            for name in MIGRATIONS:
                path = scripts_dir / name
                sql = path.read_text(encoding="utf-8")
                digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                if name in known:
                    if known[name] != digest:
                        raise RuntimeError(f"migration checksum mismatch: {name}")
                    continue
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations(name, sha256) VALUES (%s, %s)",
                    (name, digest),
                )
                applied.append(name)
        if mode == "dry-run":
            conn.rollback()
        else:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return applied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("apply", "dry-run"))
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "scripts",
    )
    args = parser.parse_args()
    changed = run(args.mode, args.scripts_dir)
    action = "validated then rolled back" if args.mode == "dry-run" else "applied"
    print(f"migrations {action}: {', '.join(changed) if changed else 'none pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
