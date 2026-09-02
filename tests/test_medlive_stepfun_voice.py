import json
from types import SimpleNamespace

from livekit import rtc

from medlive.agent.providers import _build_stt, _build_tts
from medlive.agent.stepfun_stt import (
    StepFunASR,
    StepFunStreamingSTT,
    _read_asr_sse,
)
from medlive.agent.stepfun_tts import StepFunTTS, _stepfun_tts_url
from medlive.config.settings import (
    VoiceSettings,
    _env_voice_settings,
    public_model_options,
    validate_voice_config_selection,
)


def test_stepfun_asr_builds_frozen_pcm_request():
    provider = StepFunASR(api_key="test-step-key")
    frame = rtc.AudioFrame(
        data=bytes(range(32)),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=16,
    )

    payload = provider._request_payload(frame, language="zh")

    audio = payload["audio"]
    assert audio["data"]
    assert audio["input"]["transcription"] == {
        "language": "zh",
        "model": "stepaudio-2.5-asr",
        "enable_itn": True,
        "enable_timestamp": False,
    }
    assert audio["input"]["format"] == {
        "type": "pcm",
        "codec": "pcm_s16le",
        "rate": 16000,
        "bits": 16,
        "channel": 1,
    }


async def test_stepfun_asr_sse_prefers_final_transcript():
    async def content():
        events = [
            {
                "type": "transcript.text.delta",
                "meta": {"session_id": "sse_1"},
                "delta": "增量错误稿",
            },
            {
                "type": "transcript.text.done",
                "meta": {"session_id": "sse_1"},
                "text": "最终识别文本",
            },
        ]
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

    response = SimpleNamespace(content=content())
    text, request_id = await _read_asr_sse(response)

    assert text == "最终识别文本"
    assert request_id == "sse_1"


def test_stepfun_tts_url_uses_step_plan_model_query():
    url = _stepfun_tts_url(
        "https://api.stepfun.com/step_plan/v1/realtime/audio",
        "stepaudio-2.5-tts",
    )
    assert url.startswith("wss://api.stepfun.com/step_plan/v1/realtime/audio?")
    assert "model=stepaudio-2.5-tts" in url


def test_stepfun_environment_maps_one_key_to_stt_and_tts(monkeypatch):
    monkeypatch.setenv("VOICE_STT_PROVIDER", "stepfun")
    monkeypatch.setenv("VOICE_TTS_PROVIDER", "stepfun")
    monkeypatch.setenv("STEPFUN_API_KEY", "step-plan-secret")
    monkeypatch.setenv(
        "VOICE_LLM_BASE_URL",
        "https://api.stepfun.com/step_plan/v1",
    )
    monkeypatch.setenv("VOICE_LLM_MODEL", "step-3.5-flash")
    monkeypatch.delenv("VOICE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("VOICE_STT_MODEL", raising=False)
    monkeypatch.delenv("VOICE_TTS_MODEL", raising=False)
    monkeypatch.delenv("VOICE_TTS_VOICE", raising=False)
    monkeypatch.delenv("VOICE_TTS_API_KEY", raising=False)
    monkeypatch.delenv("VOICE_TTS_BASE_URL", raising=False)

    voice = _env_voice_settings()

    assert voice.stt_provider == "stepfun"
    assert voice.stt_model == "stepaudio-2.5-asr"
    assert voice.stt_access_token == "step-plan-secret"
    assert voice.stt_base_url == "https://api.stepfun.com/step_plan/v1"
    assert voice.llm_model == "step-3.5-flash"
    assert voice.llm_api_key == "step-plan-secret"
    assert voice.tts_provider == "stepfun"
    assert voice.tts_model == "stepaudio-2.5-tts"
    assert voice.tts_voice == "cixingnansheng"
    assert voice.tts_api_key == "step-plan-secret"


async def test_stepfun_provider_builders_keep_separate_safety_pipeline():
    voice = VoiceSettings(
        stt_provider="stepfun",
        stt_access_token="step-plan-secret",
        stt_model="stepaudio-2.5-asr",
        stt_base_url="https://api.stepfun.com/step_plan/v1",
        tts_provider="stepfun",
        tts_model="stepaudio-2.5-tts",
        tts_voice="cixingnansheng",
        tts_api_key="step-plan-secret",
        tts_base_url=("wss://api.stepfun.com/step_plan/v1/realtime/audio"),
    )
    fake_vad = SimpleNamespace()
    settings = SimpleNamespace(voice=voice)

    stt_provider = _build_stt(settings, fake_vad)
    tts_provider = _build_tts(settings)

    assert isinstance(stt_provider, StepFunStreamingSTT)
    assert isinstance(stt_provider.wrapped_stt, StepFunASR)
    assert isinstance(tts_provider, StepFunTTS)
    await stt_provider.aclose()
    await tts_provider.aclose()


def test_stepfun_is_available_to_runtime_configuration():
    options = public_model_options()
    assert "stepfun" in {item["provider"] for item in options["stt"]["providers"]}
    assert "stepfun" in {item["provider"] for item in options["tts"]["providers"]}

    validate_voice_config_selection(
        {
            "voice": {
                "stt": {
                    "provider": "stepfun",
                    "model": "stepaudio-2.5-asr",
                },
                "tts": {
                    "provider": "stepfun",
                    "model": "stepaudio-2.5-tts",
                    "voice": "cixingnansheng",
                },
            }
        }
    )
