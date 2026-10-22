"""Regression coverage for grounded facts, revisions, persistence and runtime budgets."""
from datetime import datetime, timedelta
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from medrag.memory import AtomicFact, MemorySystem
from medrag.memory.extraction import extract_rules, extract_llm
from medrag.memory.long_term import LongTermMemory
from medrag.memory.schema import ContextAssembler, ContextBudgetError, RuntimeContextSchema, estimate_tokens
from medrag.memory.types import ConsolidationConfig, MemoryItem


@pytest.fixture
def memory():
    instance = MemorySystem(async_extraction=False)
    yield instance
    instance.close()


def fact(value="青霉素", predicate="过敏史", **kwargs):
    return AtomicFact(subject="user", predicate=predicate, value=value,
                      evidence=f"我对{value}过敏", source_message_id="m1", **kwargs)


@pytest.mark.parametrize("text", [
    "青霉素过敏是什么意思？", "我对青霉素过敏？", "我是不是有高血压", "如果我有高血压",
    "假设我对青霉素过敏，血压150/95", "建议我长期服用某药", "我朋友有高血压", "我想了解高血压",
])
def test_questions_hypotheticals_and_third_parties_do_not_create_facts(memory, text):
    memory.add_message("user", text)
    assert memory.long_term.count() == 0


def test_only_asserted_clause_is_kept(memory):
    memory.add_message("user", "我有高血压，能吃什么药？")
    assert [(f.predicate, f.value) for f in memory.preferences.facts()] == [("慢性病史", "高血压")]
    assert all("能吃" not in x.content for x in memory.long_term.items)


def test_atomic_split_provenance_and_sync_visibility(memory):
    memory.add_message("user", "我对青霉素和头孢过敏，有高血压")
    facts = memory.preferences.facts()
    assert len(facts) == 3
    assert set(memory.preferences.values("过敏史")) == {"青霉素", "头孢"}
    assert len({f.source_message_id for f in facts}) == 1
    assert all(f.recorded_at and f.evidence and f.subject == "user" for f in facts)
    assert all(f.effective_at is None and f.confirmation_status == "self_reported" for f in facts)


def test_assistant_reply_never_becomes_profile_or_ltm(memory):
    memory.store_assistant_reply("患者有高血压，建议长期用药")
    assert not memory.long_term.items
    assert len(memory.short_term) == 1


def test_negation_creates_revision_not_semantic_duplicate(memory):
    memory.add_message("user", "我对青霉素过敏")
    memory.add_message("user", "我对青霉素不过敏")
    history = memory.preferences.facts("过敏史", include_history=True)
    assert len(history) == 2
    assert history[0].status == "superseded"
    assert history[1].negated and history[1].supersedes == [history[0].fact_id]
    assert memory.preferences.values("过敏史") == ["否认：青霉素"]
    assert not any(x.fact and not x.fact.negated for x in memory.recall("青霉素"))


def test_different_allergens_never_merge_even_with_identical_vectors():
    ltm = LongTermMemory()
    ltm.set_consolidation_config(ConsolidationConfig())
    ltm.store_fact(fact(), embedding=np.array([1., 0.]))
    second = fact("头孢"); second.source_message_id = "m2"
    ltm.store_fact(second, embedding=np.array([1., 0.]))
    ltm.consolidate()
    assert ltm.count() == 2


def test_vitals_keep_observation_history_and_distinct_metrics(memory):
    memory.add_message("user", "我血压150/95，血糖6.2")
    memory.add_message("user", "我血压130/85")
    assert memory.preferences.values("血压") == ["150/95", "130/85"]
    assert memory.preferences.values("血糖") == ["6.2"]
    assert memory.preferences.get("血压") == "130/85"
    assert [f.value for f in memory.preferences.current_facts() if f.predicate == "血压"] == ["130/85"]
    assert len(memory.preferences.facts("血压", include_history=True)) == 2


def test_family_history_has_its_own_subject(memory):
    memory.add_message("user", "我父亲有高血压")
    f = memory.preferences.facts()[0]
    assert (f.subject, f.predicate, f.value) == ("父亲", "家族史", "高血压")
    assert not memory.preferences.values("慢性病史")


def test_delayed_old_enrichment_does_not_resurrect_corrected_fact():
    ltm = LongTermMemory()
    old = fact(recorded_at=datetime(2026, 1, 1))
    correction = fact(negated=True, operation="update", recorded_at=datetime(2026, 1, 2))
    correction.source_message_id = "m2"
    ltm.store_fact(correction)
    ltm.store_fact(old)
    assert old.status == "superseded"
    assert [x.fact for x in ltm.recall("青霉素")] == [correction]


def test_clear_cancels_late_enrichment():
    started, release = Event(), Event()
    def extract(*args):
        started.set()
        assert release.wait(3)
        return [fact()]
    with patch("medrag.memory.extraction.extract_llm", side_effect=extract):
        memory = MemorySystem()
        memory.add_message("user", "我希望记录健康信息")
        assert started.wait(3)
        memory.clear()
        release.set()
        assert memory.wait_for_extraction(3)
        assert memory.long_term.count() == 0
        memory.close()


def test_fact_json_roundtrip_with_revisions_and_decay(tmp_path):
    path = str(tmp_path / "memory.json")
    memory = MemorySystem(persist_path=path, async_extraction=False)
    memory.add_message("user", "我对青霉素过敏")
    memory.add_message("user", "我对青霉素不过敏")
    restored = MemorySystem(persist_path=path, async_extraction=False)
    assert restored.preferences.values("过敏史") == ["否认：青霉素"]
    assert len(restored.preferences.facts(include_history=True)) == 2
    assert restored.long_term.items[0].base_importance == .8


def test_decay_idempotent_and_single_item_expires(tmp_path):
    ltm = LongTermMemory(persist_path=str(tmp_path / "memory.json"))
    ltm.set_consolidation_config(ConsolidationConfig(ttl_days=1000))
    ltm.store("独立记录", .8)
    created = datetime(2026, 1, 1)
    ltm.items[0].created_at = ltm.items[0].decay_anchor = created
    now = created+timedelta(days=30)
    ltm.consolidate(now=now)
    first = ltm.items[0].importance
    ltm.consolidate(now=now)
    assert ltm.items[0].importance == first == pytest.approx(.8*.995**30)
    restored = LongTermMemory(persist_path=ltm._persist_path)
    restored.set_consolidation_config(ltm.consolidation_cfg)
    restored.consolidate(now=now)
    assert restored.items[0].importance == first
    restored.set_consolidation_config(ConsolidationConfig(ttl_days=30, min_importance=.7))
    result = restored.consolidate(now=created+timedelta(days=31))
    assert result.expired == 1 and restored.count() == 0


def test_optional_slots_pruned_required_facts_preserved():
    assembler = ContextAssembler(budget=64)
    assembler.add("constraints", "必须保留", 1, required=True)
    assembler.add("history", "历史"*100, 100, token_budget=32)
    text = assembler.assemble()
    assert "必须保留" in text
    assert estimate_tokens(text) <= 64
    small = ContextAssembler(budget=1)
    small.add("constraints", "必须保留", required=True)
    with pytest.raises(ContextBudgetError):
        small.assemble()


def test_per_slot_trimming_and_schema_fallback():
    assembler = ContextAssembler(budget=100)
    assembler.add("recall_memory", "长内容"*100, token_budget=24)
    assert estimate_tokens(assembler.assemble()) <= 24
    assert RuntimeContextSchema.for_mode("unknown") == RuntimeContextSchema.for_mode("chat")
    assert set(RuntimeContextSchema.for_mode("react").slots) >= {
        "constraints", "profile", "planner", "task_memory", "tool_state", "recall_memory"}


def test_llm_json_prompt_and_grounding_validation():
    import json
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = json.dumps({"facts": [
        {"subject": "user", "predicate": "过敏史", "value": "青霉素", "evidence": "我对青霉素过敏"},
        {"subject": "user", "predicate": "慢性病史", "value": "高血压", "evidence": "伪造的来源"},
    ]})
    with patch("medrag.llm.clients.get_llm_client", return_value=client), \
         patch("medrag.llm.provider.is_llm_provider_configured", return_value=True), \
         patch("medrag.llm.provider.get_llm_provider", return_value=MagicMock(default_model="mock")):
        facts = extract_llm("我对青霉素过敏", "message-1")
    assert len(facts) == 1 and facts[0].value == "青霉素"
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert '{"facts"' in messages[0]["content"]  # JSON braces no longer pass through str.format


def test_pg_sync_includes_metadata_and_retries_on_failure():
    with patch("medrag.infrastructure.storage.postgres_client.ltm_load_all", return_value=[]), \
         patch("medrag.infrastructure.storage.postgres_client.ltm_sync_items", side_effect=[RuntimeError("offline"), [101, 102]]) as sync:
        ltm = LongTermMemory(username="alice")
        ltm.store_fact(fact())
        assert len(ltm._dirty) == 1
        newer = fact("头孢"); newer.source_message_id = "m2"
        ltm.store_fact(newer)
        assert not ltm._dirty
        assert [x.id for x in ltm.items] == [101, 102]
        username, rows, deleted = sync.call_args.args
        assert username == "alice" and all(row["fact"]["source_role"] == "user" for row in rows)


def test_legacy_memory_loads_without_inventing_fact_provenance(tmp_path):
    import json
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"items": [{"id": 3, "content": "历史原文", "importance": .5}]}))
    ltm = LongTermMemory(persist_path=str(path))
    assert ltm.items[0].fact is None


def test_model_budget_rebuilt_for_new_observations():
    from medrag.react.engine import ReActEngine
    engine = ReActEngine(MagicMock(), token_counter=len, context_window=6000, output_reserve=1000)
    messages = engine._build_initial_messages("当前问题", "过敏史：青霉素")
    messages += [{"role": "assistant", "content": "行动：检索"},
                 {"role": "system", "content": "Observation: "+"资料"*10000}]
    bounded = engine._budget_messages(messages)
    assert engine.last_context_usage["input"]+1000 <= 6000
    assert "过敏史：青霉素" in bounded[0]["content"]
    assert "当前问题" == bounded[1]["content"]
    assert "规则" in bounded[0]["content"]


def test_explicit_vital_correction_keeps_old_version(memory):
    memory.add_message("user", "我血压150/95")
    memory.add_message("user", "我血压不是150/95，是130/85")
    history = memory.preferences.facts("血压", include_history=True)
    assert len(history) == 2
    assert history[0].status == "superseded"
    assert history[1].value == "130/85"
    assert history[1].supersedes == [history[0].fact_id]


def test_explicit_allergen_correction_keeps_other_values(memory):
    memory.add_message("user", "我对青霉素和花生过敏")
    memory.add_message("user", "我不是对青霉素过敏，是对头孢过敏")
    assert set(memory.preferences.values("过敏史")) == {"花生", "否认：青霉素", "头孢"}
    correction = [f for f in memory.preferences.facts() if f.value == "头孢"][0]
    assert correction.replaces_value == "青霉素"


def test_mixed_negation_is_scoped_to_each_statement(memory):
    memory.add_message("user", "我没有高血压但有糖尿病")
    assert {(f.value, f.negated) for f in memory.preferences.facts()} == {("高血压", True), ("糖尿病", False)}


def test_relative_subject_continues_until_explicit_user_statement(memory):
    memory.add_message("user", "我父亲有高血压，也有糖尿病，我有哮喘")
    facts = memory.preferences.facts()
    assert {(f.subject, f.value) for f in facts} == {("父亲", "高血压"), ("父亲", "糖尿病"), ("user", "哮喘")}


def test_stop_directive_survives_runtime_pruning():
    from medrag.react.engine import ReActEngine
    engine = ReActEngine(MagicMock(), token_counter=len, context_window=6000, output_reserve=1000)
    messages = engine._build_initial_messages("问题", "")
    messages += [{"role": "assistant", "content": "行动：检索"},
                 {"role": "system", "content": "Observation: "+"历史资料"*10000},
                 {"role": "system", "content": "立即停止工具调用，给出最终答案"}]
    bounded = engine._budget_messages(messages)
    assert "立即停止工具调用" in bounded[0]["content"]


def test_sessions_share_user_facts_but_not_short_term_windows():
    from threading import Lock
    from types import SimpleNamespace
    from medrag.service.chat_service import MedicalChatService
    service = object.__new__(MedicalChatService)
    service._fixed_memory_system = False
    service._memory_systems = {}
    service._memory_lock = Lock()
    service.hybrid_retriever = SimpleNamespace(qa=None)
    with patch("medrag.infrastructure.storage.postgres_client.ltm_load_all", return_value=[]), \
         patch("medrag.infrastructure.storage.postgres_client.pref_load_all", return_value={}), \
         patch("medrag.infrastructure.storage.postgres_client.ltm_sync_items", return_value=[101]):
        first = service._get_request_memory("alice", "session-a")
        second = service._get_request_memory("alice", "session-b")
        first._async_extraction = second._async_extraction = False
        assert first.long_term is second.long_term
        first.add_message("user", "我对青霉素过敏")
        assert second.preferences.values("过敏史") == ["青霉素"]
        assert len(second.short_term) == 0
        assert first.preferences.facts()[0].source_session_id == "session-a"
