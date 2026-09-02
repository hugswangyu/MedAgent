"""装配 LiveKit AgentSession 的在线模型提供方。"""

from __future__ import annotations

from livekit.agents import AgentSession
from livekit.plugins import minimax, openai, silero, volcengine

from medlive.agent.dashscope_tts import DashScopeRealtimeTTS
from medlive.agent.stepfun_stt import StepFunASR, StepFunStreamingSTT
from medlive.agent.stepfun_tts import StepFunTTS
from medlive.config.settings import AppSettings


def build_agent_session(settings: AppSettings) -> AgentSession:
    """创建实时语音会话，保留当前线上链路调优参数。"""

    voice = settings.voice
    vad = silero.VAD.load()
    return AgentSession(
        stt=_build_stt(settings, vad),
        llm=openai.LLM(
            model=voice.llm_model,
            base_url=voice.llm_base_url,
            api_key=voice.llm_api_key,
        ),
        tts=_build_tts(settings),
        preemptive_generation=True,
        min_interruption_duration=0.2,
        min_endpointing_delay=0.0,
        max_endpointing_delay=0.05,
        turn_detection="stt",
        vad=vad,
    )


def _build_stt(settings: AppSettings, vad):
    """按配置创建 STT；StepFun 用同一 VAD 完成准实时切句。"""

    voice = settings.voice
    if voice.stt_provider == "stepfun":
        return StepFunStreamingSTT(
            stt=StepFunASR(
                api_key=voice.stt_access_token,
                model=voice.stt_model,
                base_url=voice.stt_base_url,
            ),
            vad=vad,
        )
    if voice.stt_provider != "volcengine_bigmodel":
        raise ValueError(f"不支持的 STT provider: {voice.stt_provider}")
    return volcengine.BigModelSTT(
        app_id=voice.stt_app_id,
        access_token=voice.stt_access_token,
        model_name=voice.stt_model,
        enable_itn=False,
        enable_punc=False,
        enable_ddc=False,
        vad_segment_duration=1200,
        end_window_size=240,
        force_to_speech_time=1000,
        interim_results=True,
    )


def _build_tts(settings: AppSettings):
    """根据运行时配置选择 TTS provider。"""

    voice = settings.voice
    if voice.tts_provider == "stepfun":
        return StepFunTTS(
            model=voice.tts_model,
            voice=voice.tts_voice,
            api_key=voice.tts_api_key,
            base_url=voice.tts_base_url,
            sample_rate=24000,
            speed_ratio=1.05,
        )
    if voice.tts_provider in {"dashscope", "dashscope_realtime", "qwen_realtime"}:
        return DashScopeRealtimeTTS(
            model=voice.tts_model,
            voice=voice.tts_voice,
            api_key=voice.tts_api_key,
            base_url=voice.tts_base_url,
            sample_rate=24000,
            speech_rate=1.05,
        )

    return minimax.TTS(
        model=voice.tts_model,
        voice=voice.tts_voice,
        api_key=voice.tts_api_key,
        base_url=voice.tts_base_url,
        audio_format="pcm",
        sample_rate=24000,
        speed=1.05,
    )
