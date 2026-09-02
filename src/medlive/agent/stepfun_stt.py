"""StepFun Step Plan ASR 的 LiveKit 适配器。"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any

import aiohttp
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    NotGivenOr,
    stt,
    utils,
)

DEFAULT_STEPFUN_BASE_URL = "https://api.stepfun.com/step_plan/v1"


@dataclass(frozen=True)
class StepFunASROptions:
    """StepFun ASR 运行参数。"""

    model: str
    api_key: str
    base_url: str
    sample_rate: int = 16000
    language: str = "zh"


class StepFunASR(stt.STT):
    """通过 Step Plan HTTP+SSE 接口识别一个完整语音片段。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "stepaudio-2.5-asr",
        base_url: str = DEFAULT_STEPFUN_BASE_URL,
        sample_rate: int = 16000,
        language: str = "zh",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("StepFun ASR api_key 必须配置")
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                diarization=False,
            )
        )
        self._opts = StepFunASROptions(
            model=model,
            api_key=api_key.strip(),
            base_url=base_url.rstrip("/"),
            sample_rate=sample_rate,
            language=language,
        )
        self._session = http_session
        self._owns_session = http_session is None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "StepFun"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        request_language = language if isinstance(language, str) else self._opts.language
        payload = self._request_payload(frame, language=request_language)
        session = await self._http_session()
        try:
            async with session.post(
                f"{self._opts.base_url}/audio/asr/sse",
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise APIStatusError(
                        "StepFun ASR 请求失败",
                        status_code=response.status,
                        body=body[:1000],
                    )
                text, request_id = await _read_asr_sse(response)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise APITimeoutError("StepFun ASR 请求超时") from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError(f"StepFun ASR 连接失败: {type(exc).__name__}") from exc

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id or f"stepfun_asr_{uuid.uuid4().hex}",
            alternatives=[
                stt.SpeechData(
                    language=request_language,
                    text=text.strip(),
                    confidence=1.0 if text.strip() else 0.0,
                )
            ],
        )

    def _request_payload(self, frame: rtc.AudioFrame, *, language: str) -> dict[str, Any]:
        """生成 Step Plan ASR 冻结请求体。"""

        return {
            "audio": {
                "data": base64.b64encode(bytes(frame.data)).decode("ascii"),
                "input": {
                    "transcription": {
                        "language": language,
                        "model": self._opts.model,
                        "enable_itn": True,
                        "enable_timestamp": False,
                    },
                    "format": {
                        "type": "pcm",
                        "codec": "pcm_s16le",
                        "rate": frame.sample_rate,
                        "bits": 16,
                        "channel": frame.num_channels,
                    },
                },
            }
        }

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
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None


class StepFunStreamingSTT(stt.StreamAdapter):
    """使用 LiveKit VAD 将 StepFun 片段识别适配为流式 STT。"""

    async def aclose(self) -> None:
        wrapped = self.wrapped_stt
        await super().aclose()
        await wrapped.aclose()


async def _read_asr_sse(
    response: aiohttp.ClientResponse,
) -> tuple[str, str]:
    """读取增量 SSE，并优先返回 done 事件中的完整文本。"""

    deltas: list[str] = []
    final_text = ""
    request_id = ""
    async for raw_line in response.content:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type") or "")
        meta = event.get("meta")
        if isinstance(meta, dict):
            request_id = str(meta.get("session_id") or request_id)
        if event_type == "transcript.text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "transcript.text.done":
            final_text = str(event.get("text") or "")
        elif event_type == "error":
            raise APIStatusError(
                str(event.get("message") or "StepFun ASR 返回错误"),
                body=event,
            )
    return final_text or "".join(deltas), request_id
