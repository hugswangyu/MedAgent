import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from medrag.contracts.phase0 import VoiceTurnRecordRequest


def test_phase2_migration_adds_restricted_raw_and_final_tts_fields():
    migration = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "phase2_safe_voice.sql"
    ).read_text(encoding="utf-8")

    assert "raw_model_text" in migration
    assert "final_tts_text" in migration
    assert "idx_voice_turns_turn_id" in migration


def test_worker_scope_allows_turn_write():
    from medrag.app.auth_manager import WORKER_SCOPES

    assert "voice:turn:write" in WORKER_SCOPES


def test_turn_endpoint_associates_messages_safety_and_evidence(monkeypatch):
    from medrag.app.api import internal_v1

    captured = {}
    monkeypatch.setattr(
        internal_v1,
        "_authorize",
        lambda **kwargs: (None, SimpleNamespace(user_id="user_1")),
    )
    monkeypatch.setattr(
        internal_v1.phase1_repository,
        "record_voice_turn",
        lambda **kwargs: captured.update(kwargs),
    )
    payload = VoiceTurnRecordRequest(
        session_id="vs_1",
        turn_id="turn_1",
        idempotency_key="phase2-turn-write-1",
        turn_index=1,
        user_text="我现在胸痛",
        raw_model_text="",
        final_text="请立即拨打120。",
        input_safety={"action": "emergency"},
        output_safety=[],
        evidence=[],
    )

    response = asyncio.run(
        internal_v1.record_voice_turn(
            "vs_1",
            payload,
            authorization="Bearer test",
            x_request_nonce="00000000-0000-0000-0000-000000000001",
        )
    )
    body = json.loads(response.body)

    assert body["status"] == "ok"
    assert body["data"]["turn_id"] == "turn_1"
    assert captured["turn_id"] == "turn_1"
    assert captured["user_text"] == "我现在胸痛"
    assert captured["final_text"] == "请立即拨打120。"
    assert captured["safety_result"]["input"]["action"] == "emergency"
