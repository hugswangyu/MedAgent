"""DashScope Qwen realtime TTS provider for LiveKit."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    tts,
    utils,
)

DEFAULT_DASHSCOPE_REALTIME_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


@dataclass(frozen=True)
class DashScopeRealtimeTTSOptions:
    """Runtime options for DashScope realtime TTS."""

    model: str
    voice: str
    api_key: str
    base_url: str
    sample_rate: int = 24000
    speech_rate: float = 1.05
    language_type: str = "Chinese"


class DashScopeRealtimeTTS(tts.TTS):
    """LiveKit TTS wrapper for DashScope qwen3-tts-flash-realtime."""

    def __init__(
        self,
        *,
        model: str = "qwen3-tts-flash-realtime",
        voice: str = "Cherry",
        api_key: str,
        base_url: str = DEFAULT_DASHSCOPE_REALTIME_URL,
        sample_rate: int = 24000,
        speech_rate: float = 1.05,
        language_type: str = "Chinese",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DashScope TTS api_key is required")
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = DashScopeRealtimeTTSOptions(
            model=model,
            voice=voice,
            api_key=api_key,
            base_url=base_url,
            sample_rate=sample_rate,
            speech_rate=speech_rate,
            language_type=language_type,
        )
        self._session = http_session
        self._owns_session = http_session is None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_lock = asyncio.Lock()
        self._streams: set[DashScopeRealtimeSynthesizeStream] = set()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "DashScope"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        stream = DashScopeRealtimeSynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        if self._ws is not None and not self._ws.closed:
            return self._ws

        if self._session is None:
            try:
                self._session = utils.http_context.http_session()
                self._owns_session = False
            except RuntimeError:
                self._session = aiohttp.ClientSession()
                self._owns_session = True

        self._ws = await asyncio.wait_for(
            self._session.ws_connect(
                _dashscope_realtime_url(self._opts.base_url, self._opts.model),
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
            ),
            timeout=timeout,
        )
        return self._ws

    async def _close_ws(self) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._close_ws()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None


class DashScopeRealtimeSynthesizeStream(tts.SynthesizeStream):
    """A single LiveKit TTS stream backed by one DashScope realtime response."""

    def __init__(self, *, tts: DashScopeRealtimeTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: DashScopeRealtimeTTS = tts
        self._opts = tts._opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = f"dashscope_{uuid.uuid4().hex}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
            stream=True,
        )

        async with self._tts._ws_lock:
            try:
                ws = await self._tts._connect_ws(self._conn_options.timeout)
                await self._send_session_update(ws)
                await self._run_response(ws, output_emitter, request_id)
            except TimeoutError as exc:
                await self._tts._close_ws()
                raise APITimeoutError("DashScope realtime TTS request timed out") from exc
            except asyncio.TimeoutError as exc:
                await self._tts._close_ws()
                raise APITimeoutError("DashScope realtime TTS request timed out") from exc
            except APIError:
                await self._tts._close_ws()
                raise
            except Exception as exc:
                await self._tts._close_ws()
                raise APIConnectionError(
                    f"DashScope realtime TTS failed: {type(exc).__name__}: {exc}"
                ) from exc

    async def _send_session_update(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await _send_json(
            ws,
            {
                "event_id": _event_id(),
                "type": "session.update",
                "session": {
                    "voice": self._opts.voice,
                    "mode": "commit",
                    "response_format": "pcm",
                    "sample_rate": self._opts.sample_rate,
                    "speech_rate": self._opts.speech_rate,
                    "language_type": self._opts.language_type,
                },
            },
        )

    async def _run_response(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        fallback_request_id: str,
    ) -> None:
        response_done = asyncio.Event()
        input_done = asyncio.Event()
        committed = False
        segment_started = False

        async def input_task() -> None:
            nonlocal committed
            saw_text = False
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    if saw_text and not committed:
                        self._mark_started()
                        await _send_json(
                            ws,
                            {"event_id": _event_id(), "type": "input_text_buffer.commit"},
                        )
                        committed = True
                    continue

                if data:
                    saw_text = True
                    await _send_json(
                        ws,
                        {
                            "event_id": _event_id(),
                            "type": "input_text_buffer.append",
                            "text": data,
                        },
                    )
            input_done.set()
            if not committed:
                response_done.set()

        async def recv_task() -> None:
            nonlocal segment_started
            while not response_done.is_set():
                if input_done.is_set() and not committed:
                    response_done.set()
                    return

                msg = await asyncio.wait_for(ws.receive(), timeout=self._conn_options.timeout)
                if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING}:
                    raise APIConnectionError("DashScope realtime TTS websocket closed unexpectedly")
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                payload = json.loads(str(msg.data))
                msg_type = str(payload.get("type") or "")
                if msg_type == "response.created":
                    response_id = str(payload.get("response", {}).get("id") or fallback_request_id)
                    if not segment_started:
                        output_emitter.start_segment(segment_id=response_id)
                        segment_started = True
                elif msg_type == "response.audio.delta":
                    if not segment_started:
                        output_emitter.start_segment(segment_id=fallback_request_id)
                        segment_started = True
                    audio_delta = payload.get("delta")
                    if isinstance(audio_delta, str) and audio_delta:
                        output_emitter.push(base64.b64decode(audio_delta))
                elif msg_type == "response.done":
                    if segment_started:
                        output_emitter.end_input()
                    response_done.set()
                elif msg_type in {"error", "response.error"}:
                    raise APIError(f"DashScope realtime TTS error: {payload}")

        tasks = [asyncio.create_task(input_task()), asyncio.create_task(recv_task())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)


async def _send_json(ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]) -> None:
    await ws.send_str(json.dumps(payload, ensure_ascii=False))


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _dashscope_realtime_url(base_url: str, model: str) -> str:
    """Build DashScope realtime websocket URL with model query parameter."""

    url = (base_url or DEFAULT_DASHSCOPE_REALTIME_URL).strip().rstrip("/")
    if url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    elif url.startswith("https://"):
        url = "wss://" + url[len("https://") :]

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["model"] = model
    return urlunparse(parsed._replace(query=urlencode(query)))
