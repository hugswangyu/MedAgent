
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from livekit.agents.metrics import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics

from medlive.logging.events import EventLogger


@dataclass
class MetricsState:
    """指标回调之间共享的会话状态。"""

    last_user_created_at: float | None = None
    last_user_stop_at: float | None = None
    last_interim_text: str = ""
    last_tcp_rtt_ms: float | None = None


def register_session_metrics_hooks(
    session: Any,
    logger: logging.Logger,
    metrics_logger: EventLogger,
    state: MetricsState,
) -> None:
    """注册会话指标回调。"""

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: Any) -> None:
        """记录用户文本到助手首段音频的端到端延迟。"""
        item = getattr(ev, "item", None)
        if item is None or getattr(item, "type", None) != "message":
            return

        role = getattr(item, "role", None)
        if role == "user":
            state.last_user_created_at = getattr(item, "created_at", None)
            return

        if role == "assistant" and state.last_user_created_at is not None:
            item_metrics = getattr(item, "metrics", {}) or {}
            started = item_metrics.get("started_speaking_at")
            if isinstance(started, (int, float)):
                e2e_from_user_ms = (started - state.last_user_created_at) * 1000.0
                payload = {
                    "e2e_ms": round(e2e_from_user_ms, 1),
                    "tcp_rtt_ms": round(state.last_tcp_rtt_ms, 1)
                    if state.last_tcp_rtt_ms
                    else None,
                }
                logger.info("latency.e2e_user_to_first_audio", extra=payload)
                metrics_logger.append("latency.e2e_user_to_first_audio", payload)

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: Any) -> None:
        """记录用户停止说话的时间戳。"""
        if (
            getattr(ev, "old_state", None) == "speaking"
            and getattr(ev, "new_state", None) == "listening"
        ):
            state.last_user_stop_at = float(getattr(ev, "created_at", time.time()))

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: Any) -> None:
        """记录中间/最终转写，以及最终转写收敛延迟。"""
        transcript = str(getattr(ev, "transcript", "") or "").strip()
        if not transcript:
            return

        is_final = bool(getattr(ev, "is_final", False))
        if not is_final:
            if transcript != state.last_interim_text:
                state.last_interim_text = transcript
                payload = {
                    "text": transcript,
                    "text_len": len(transcript),
                    "is_final": False,
                }
                logger.info("transcript.interim", extra=payload)
                metrics_logger.append("transcript.interim", payload)
            return

        now_ts = float(getattr(ev, "created_at", time.time()))
        payload_final = {
            "text": transcript,
            "text_len": len(transcript),
            "is_final": True,
            "backfill_from_interim": bool(state.last_interim_text),
        }
        logger.info("transcript.final", extra=payload_final)
        metrics_logger.append("transcript.final", payload_final)
        state.last_interim_text = ""

        if state.last_user_stop_at is None:
            return

        payload = {
            "finalization_ms": round((now_ts - state.last_user_stop_at) * 1000.0, 1),
            "transcript_len": len(transcript),
            "is_final": True,
        }
        logger.info("latency.stt_final", extra=payload)
        metrics_logger.append("latency.stt_final", payload)
        state.last_user_stop_at = None

    @session.on("metrics_collected")
    def _on_metrics(ev: Any) -> None:
        """处理 LiveKit 上报的 EOU、LLM、TTS 和 STT 指标。"""
        m = getattr(ev, "metrics", None)
        if m is None:
            return

        if isinstance(m, EOUMetrics):
            payload = {
                "end_of_utterance_ms": round(m.end_of_utterance_delay * 1000.0, 1),
                "transcription_ms": round(m.transcription_delay * 1000.0, 1),
                "on_user_turn_completed_ms": round(
                    m.on_user_turn_completed_delay * 1000.0, 1
                ),
                "speech_id": m.speech_id,
            }
            logger.info("latency.eou", extra=payload)
            metrics_logger.append("latency.eou", payload)
        elif isinstance(m, LLMMetrics):
            payload = {
                "ttft_ms": round(m.ttft * 1000.0, 1),
                "duration_ms": round(m.duration * 1000.0, 1),
                "request_id": m.request_id,
                "model": m.label,
            }
            logger.info("latency.llm", extra=payload)
            metrics_logger.append("latency.llm", payload)
        elif isinstance(m, TTSMetrics):
            payload = {
                "ttfb_ms": round(m.ttfb * 1000.0, 1),
                "duration_ms": round(m.duration * 1000.0, 1),
                "audio_duration_ms": round(m.audio_duration * 1000.0, 1),
                "request_id": m.request_id,
                "model": m.label,
            }
            logger.info("latency.tts", extra=payload)
            metrics_logger.append("latency.tts", payload)
        elif isinstance(m, STTMetrics):
            payload = {
                "duration_ms": round(m.duration * 1000.0, 1),
                "audio_duration_ms": round(m.audio_duration * 1000.0, 1),
                "request_id": m.request_id,
                "model": m.label,
                "streamed": m.streamed,
            }
            logger.info("latency.stt", extra=payload)
            metrics_logger.append("latency.stt", payload)


async def probe_tcp_rtt(host: str, port: int, timeout_s: float = 2.0) -> float | None:
    """估算 TCP 建连 RTT，单位为毫秒。"""
    start = time.perf_counter()
    try:
        connect_coro = asyncio.open_connection(host, port)
        _, writer = await asyncio.wait_for(connect_coro, timeout=timeout_s)
        writer.close()
        await writer.wait_closed()
        return (time.perf_counter() - start) * 1000.0
    except Exception:
        return None


def start_network_probe_task(
    *,
    livekit_url: str,
    state: MetricsState,
    logger: logging.Logger,
    metrics_logger: EventLogger,
    interval_s: float = 5.0,
) -> asyncio.Task[None] | None:
    """启动后台任务，周期性记录 TCP RTT。"""
    parsed = urlparse(livekit_url)
    probe_host = parsed.hostname
    if not probe_host:
        return None

    if parsed.port:
        probe_port = parsed.port
    elif parsed.scheme == "wss":
        probe_port = 443
    else:
        probe_port = 80

    async def _network_probe_loop() -> None:
        """循环探测 TCP RTT，直到任务被取消。"""
        while True:
            rtt = await probe_tcp_rtt(probe_host, probe_port, timeout_s=2.0)
            state.last_tcp_rtt_ms = rtt
            if rtt is not None:
                payload = {
                    "host": probe_host,
                    "port": probe_port,
                    "tcp_rtt_ms": round(rtt, 1),
                }
                logger.info("latency.network_probe", extra=payload)
                metrics_logger.append("latency.network_probe", payload)
            await asyncio.sleep(interval_s)

    return asyncio.create_task(_network_probe_loop())
