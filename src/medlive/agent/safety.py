"""语音输入旁路和 TTS 前句级安全缓冲。"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from medlive.agent.tool.medical_client import MedicalCapabilityClient

INPUT_SAFETY_FALLBACK = (
    "医疗安全检查暂时不可用，本轮不会继续生成回答。"
    "如果你有正在发生的胸痛、呼吸困难、意识异常或其他紧急症状，"
    "请立即拨打120或前往急诊。"
)
OUTPUT_SAFETY_FALLBACK = (
    "医疗输出安全检查暂时不可用，这段内容不会播报。"
    "涉及诊断、用药或急症时请咨询医生；如有紧急症状请立即拨打120。"
)
_SENTENCE_END = re.compile(r"[。！？!?；;\n]")


async def checked_tts_text(
    text: AsyncIterable[str],
    *,
    client: MedicalCapabilityClient,
    turn_id: str,
    evidence: list[dict[str, Any]],
    max_chars: int = 120,
) -> AsyncIterator[str]:
    """完整句或安全长度缓冲后检查；异常时绝不透传原文。"""

    async for sentence in sentence_segments(text, max_chars=max_chars):
        yield await checked_output_text(
            sentence,
            client=client,
            turn_id=turn_id,
            evidence=evidence,
        )


async def checked_output_text(
    text: str,
    *,
    client: MedicalCapabilityClient,
    turn_id: str,
    evidence: list[dict[str, Any]],
) -> str:
    """检查一个输出片段；不完整或自相矛盾的响应一律 fail-closed。"""

    try:
        checked = await client.output_check(
            text=text, turn_id=turn_id, evidence=evidence
        )
        allowed = checked.data.get("allowed")
        safe_text = str(checked.data.get("safe_text") or "").strip()
        if not isinstance(allowed, bool) or not safe_text:
            return OUTPUT_SAFETY_FALLBACK
        if not allowed and safe_text == text.strip():
            return OUTPUT_SAFETY_FALLBACK
        return safe_text
    except Exception:
        return OUTPUT_SAFETY_FALLBACK


async def sentence_segments(
    text: AsyncIterable[str], *, max_chars: int = 120
) -> AsyncIterator[str]:
    """把任意增量文本切成完整句；超长句按安全长度截断。"""

    buffer = ""
    async for chunk in text:
        buffer += str(chunk or "")
        while buffer:
            boundary = _first_boundary(buffer)
            if boundary is not None:
                sentence, buffer = buffer[:boundary], buffer[boundary:]
                if sentence.strip():
                    yield sentence
                continue
            if len(buffer) >= max_chars:
                sentence, buffer = buffer[:max_chars], buffer[max_chars:]
                if sentence.strip():
                    yield sentence
                continue
            break
    if buffer.strip():
        yield buffer


def collect_turn_evidence(
    records: list[dict[str, Any]], turn_index: int
) -> list[dict[str, Any]]:
    """读取当前 turn 的统一证据，忽略旧轮次和非对象值。"""

    evidence: list[dict[str, Any]] = []
    expected_turn_id = f"turn_{turn_index}"
    for record in records:
        if record.get("turn_index") != turn_index:
            continue
        raw_items = (
            record.get("unified_evidence")
            or record.get("evidence")
            or []
        )
        if isinstance(raw_items, list):
            evidence.extend(
                item
                for item in raw_items
                if isinstance(item, dict)
                and (
                    not item.get("turn_id")
                    or str(item.get("turn_id")) == expected_turn_id
                )
            )
    return evidence


def _first_boundary(text: str) -> int | None:
    match = _SENTENCE_END.search(text)
    return match.end() if match else None


def split_ready_segments(
    buffer: str, *, max_chars: int = 120, final: bool = False
) -> tuple[list[str], str]:
    """从同步缓冲中弹出完整句/安全长度片段，供 LLM 输出前置闸门使用。"""

    segments: list[str] = []
    remainder = buffer
    while remainder:
        boundary = _first_boundary(remainder)
        if boundary is not None:
            segment, remainder = remainder[:boundary], remainder[boundary:]
        elif len(remainder) >= max_chars:
            segment, remainder = remainder[:max_chars], remainder[max_chars:]
        elif final:
            segment, remainder = remainder, ""
        else:
            break
        if segment.strip():
            segments.append(segment)
    return segments, remainder
