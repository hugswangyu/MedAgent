"""Opt-in PostgreSQL checks, using only session temporary tables.

Run with MEMORY_PG_TESTS=1; requires the additive atomic-memory migration.
"""
import os
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

from medrag.memory.long_term import LongTermMemory
from medrag.memory.types import AtomicFact, ConsolidationConfig

pytestmark = pytest.mark.skipif(os.getenv("MEMORY_PG_TESTS") != "1", reason="opt-in local PostgreSQL test")


@pytest.fixture
def pg_memory(monkeypatch):
    import psycopg2
    from medrag.config.settings import settings
    from medrag.infrastructure.storage import postgres_client as storage
    conn = psycopg2.connect(host=settings.pg_host, port=settings.pg_port,
                           dbname=settings.pg_database, user=settings.pg_user,
                           password=settings.pg_password, connect_timeout=3)
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE long_term_memory (LIKE public.long_term_memory INCLUDING ALL)")
        cur.execute("ALTER TABLE pg_temp.long_term_memory ALTER COLUMN id DROP DEFAULT")
        cur.execute("ALTER TABLE pg_temp.long_term_memory ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY")
    conn.commit()

    @contextmanager
    def isolated():
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT memory_write")
        try:
            yield conn
        except Exception:
            with conn.cursor() as cur:
                cur.execute("ROLLBACK TO SAVEPOINT memory_write")
            raise
        finally:
            with conn.cursor() as cur:
                cur.execute("RELEASE SAVEPOINT memory_write")
    monkeypatch.setattr(storage, "get_conn", isolated)
    try:
        yield storage
    finally:
        conn.rollback()
        conn.close()


def test_pg_fact_revisions_and_decay_survive_reload(pg_memory):
    memory = LongTermMemory(username="test_alice")
    first = AtomicFact("user", "过敏史", "测试过敏原", "我对测试过敏原过敏", "msg1")
    memory.store_fact(first)
    second = AtomicFact("user", "过敏史", "测试过敏原", "我对测试过敏原不过敏", "msg2",
                        negated=True, operation="update")
    memory.store_fact(second)
    restored = LongTermMemory(username="test_alice")
    assert restored.items[0].fact.status == "superseded"
    assert restored.items[1].fact.supersedes == [first.fact_id]
    assert LongTermMemory(username="test_bob").count() == 0
    cfg = ConsolidationConfig(ttl_days=1000)
    restored.set_consolidation_config(cfg)
    now = datetime.utcnow()+timedelta(days=30)
    restored.consolidate(now=now)
    score = restored.items[1].importance
    again = LongTermMemory(username="test_alice")
    again.set_consolidation_config(cfg)
    again.consolidate(now=now)
    assert again.items[1].importance == pytest.approx(score)


def test_pg_rejects_cross_user_update_atomically(pg_memory):
    alice = LongTermMemory(username="test_alice")
    alice.store("original")
    row = alice.items[0].to_dict()
    first = dict(row, content="should roll back")
    missing = dict(row, id=999999)
    with pytest.raises(ValueError):
        pg_memory.ltm_sync_items("test_alice", [first, missing], [])
    assert LongTermMemory(username="test_alice").items[0].content == "original"
    with pytest.raises(ValueError):
        pg_memory.ltm_sync_items("test_bob", [dict(row, content="cross-tenant")], [])
    assert LongTermMemory(username="test_alice").items[0].content == "original"
