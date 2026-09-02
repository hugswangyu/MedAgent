"""StepFun Step Plan 流式 TTS 的 LiveKit 适配器。"""

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

DEFAULT_STEPFUN_TTS_URL = "wss://api.stepfun.com/step_plan/v1/realtime/audio"


@dataclass(frozen=True)
class StepFunTTSOptions:
    """StepFun TTS 运行参数。"""

    model: str
    voice: str
    api_key: str
    base_url: str
    sample_rate: int = 24000
    speed_ratio: float = 1.05
    instruction: str = "语气自然、清晰、克制，适合医疗信息播报"


class StepFunTTS(tts.TTS):
    """通过 Step Plan WebSocket 输出 24kHz PCM 音频。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "stepaudio-2.5-tts",
        voice: str = "cixingnansheng",
        base_url: str = DEFAULT_STEPFUN_TTS_URL,
        sample_rate: int = 24000,
        speed_ratio: float = 1.05,
        instruction: str = "语气自然、清晰、克制，适合医疗信息播报",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("StepFun TTS api_key 必须配置")
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = StepFunTTSOptions(
            model=model,
            voice=voice,
            api_key=api_key.strip(),
            base_url=base_url,
            sample_rate=sample_rate,
            speed_ratio=speed_ratio,
            instruction=instruction[:200],
        )
        self._session = http_session
        self._owns_session = http_session is None
        self._streams: set[StepFunSynthesizeStream] = set()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "StepFun"

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
        stream = StepFunSynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def _http_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            try:
                self._session = utils.http_context.http_session()
                self._owns_session = False
            except RuntimeError:
                self._session = aiohttp.ClientSession()
                self._owns_session = True
        return self._session

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None


class StepFunSynthesizeStream(tts.SynthesizeStream):
    """一个 StepFun WebSocket TTS 生成流。"""

    def __init__(self, *, tts: StepFunTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: StepFunTTS = tts
        self._opts = tts._opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = f"stepfun_tts_{uuid.uuid4().hex}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
            stream=True,
        )
        try:
            session = await self._tts._http_session()
            ws = await asyncio.wait_for(
                session.ws_connect(
                    _stepfun_tts_url(self._opts.base_url, self._opts.model),
                    headers={"Authorization": f"Bearer {self._opts.api_key}"},
                ),
                timeout=self._conn_options.timeout,
            )
            try:
                session_id = await self._create_session(ws)
                await self._run_response(ws, output_emitter, request_id, session_id)
            finally:
                await ws.close()
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise APITimeoutError("StepFun TTS 请求超时") from exc
        except APIError:
            raise
        except aiohttp.ClientError as exc:
            raise APIConnectionError(f"StepFun TTS 连接失败: {type(exc).__name__}") from exc
        finally:
            self._tts._streams.discard(self)

    async def _create_session(self, ws: aiohttp.ClientWebSocketResponse) -> str:
        while True:
            payload = await _receive_json(ws, self._conn_options.timeout)
            event_type = str(payload.get("type") or "")
            data = payload.get("data")
            data = data if isinstance(data, dict) else {}
            if event_type == "tts.connection.done":
                session_id = str(data.get("session_id") or "")
                if not session_id:
                    raise APIConnectionError("StepFun TTS 未返回 session_id")
                await _send_json(
                    ws,
                    {
                        "type": "tts.create",
                        "data": {
                            "session_id": session_id,
                            "voice_id": self._opts.voice,
                            "response_format": "pcm",
                            "sample_rate": self._opts.sample_rate,
                            "speed_ratio": self._opts.speed_ratio,
                            "text_normalization": "standard",
                            "mode": "default",
                            "instruction": self._opts.instruction,
                        },
                    },
                )
            elif event_type == "tts.response.created":
                return str(data.get("session_id") or "")
            elif event_type == "tts.response.error":
                raise APIError(str(data.get("message") or payload))

    async def _run_response(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        fallback_request_id: str,
        session_id: str,
    ) -> None:
        response_done = asyncio.Event()
        segment_started = False

        async def input_task() -> None:
            saw_text = False
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    if saw_text:
                        await _send_json(
                            ws,
                            {
                                "type": "tts.text.flush",
                                "data": {"session_id": session_id},
                            },
                        )
                    continue
                if item:
                    if not saw_text:
                        self._mark_started()
                    saw_text = True
                    await _send_json(
                        ws,
                        {
                            "type": "tts.text.delta",
                            "data": {
                                "session_id": session_id,
                                "text": item,
                            },
                        },
                    )
            await _send_json(
                ws,
                {
                    "type": "tts.text.done",
                    "data": {"session_id": session_id},
                },
            )

        async def recv_task() -> None:
            nonlocal segment_started
            while not response_done.is_set():
                payload = await _receive_json(ws, self._conn_options.timeout)
                event_type = str(payload.get("type") or "")
                data = payload.get("data")
                data = data if isinstance(data, dict) else {}
                if event_type == "tts.response.sentence.start":
                    if not segment_started:
                        segment_id = str(data.get("request_id") or fallback_request_id)
                        output_emitter.start_segment(segment_id=segment_id)
                        segment_started = True
                elif event_type == "tts.response.audio.delta":
                    if not segment_started:
                        output_emitter.start_segment(segment_id=fallback_request_id)
                        segment_started = True
                    audio = data.get("audio")
                    if isinstance(audio, str) and audio:
                        output_emitter.push(base64.b64decode(audio))
                elif event_type == "tts.response.audio.done":
                    if segment_started:
                        output_emitter.end_input()
                    response_done.set()
                elif event_type == "tts.response.error":
                    raise APIError(str(data.get("message") or payload))

        tasks = [
            asyncio.create_task(input_task()),
            asyncio.create_task(recv_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)


async def _send_json(ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]) -> None:
    await ws.send_str(json.dumps(payload, ensure_ascii=False))


async def _receive_json(ws: aiohttp.ClientWebSocketResponse, timeout: float) -> dict[str, Any]:
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
    if msg.type in {
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSING,
    }:
        raise APIConnectionError("StepFun TTS WebSocket 意外关闭")
    if msg.type != aiohttp.WSMsgType.TEXT:
        return {}
    payload = json.loads(str(msg.data))
    return payload if isinstance(payload, dict) else {}


def _stepfun_tts_url(base_url: str, model: str) -> str:
    """生成携带模型参数的 Step Plan TTS WebSocket URL。"""

    url = (base_url or DEFAULT_STEPFUN_TTS_URL).strip().rstrip("/")
    if url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    elif url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["model"] = model
    return urlunparse(parsed._replace(query=urlencode(query)))
