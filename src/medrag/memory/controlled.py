"""Phase 3 controlled episodic and medical-fact memory helpers.

These helpers are deliberately deterministic.  They only derive candidates from
user-authored text or a verified personal-document preview; model replies and
session summaries are never candidate sources.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MedicalFactCandidate:
    memory_type: str
    content: str
    structured_value: dict[str, Any]
    confidence: float


_NEGATION = re.compile(r"(?:没有|没(?:有)?|并不|不是|从未|否认|无)(?:.{0,8})(?:过敏|患有|确诊|服用|使用)")
_HISTORICAL = re.compile(r"(?:以前|曾经|小时候|去年|多年前|已经痊愈|已经好了|已停药)")
_THIRD_PERSON = re.compile(r"(?:他|她|孩子|父亲|母亲|爸爸|妈妈|朋友|同事|患者)(?:的|有|患|在|曾)")
_QUESTION = re.compile(r"(?:是什么|怎么办|如何|为什么|能否|可以吗|吗|么|？|\?)")
_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")

_USER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "allergy",
        re.compile(r"(?:我|本人)(?:对)?(?P<value>[\u4e00-\u9fffA-Za-z0-9_-]{1,30}?)(?:药物)?过敏"),
    ),
    (
        "condition",
        re.compile(r"(?:我|本人)(?:有|患有|被诊断为|确诊(?:了|为)?)(?P<value>[\u4e00-\u9fffA-Za-z0-9_-]{2,30})"),
    ),
    (
        "medication",
        re.compile(r"(?:我|本人)(?:正在|一直|目前)?(?:服用|吃|使用)(?P<value>[^，,。；;！？!?]{1,40})"),
    ),
    (
        "measurement",
        re.compile(
            r"(?:我(?:的)?|本人)(?P<label>血糖|血压|体温|尿酸|胆固醇|心率)"
            r"(?:是|为|[:：])(?P<value>[0-9./]+(?:\s*[A-Za-z%/]+)?)"
        ),
    ),
)

_DOCUMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("allergy", re.compile(r"过敏史\s*[:：]\s*(?P<value>[^，,。；;\n]{1,40})")),
    ("condition", re.compile(r"(?:诊断|疾病)\s*[:：]\s*(?P<value>[^，,。；;\n]{1,40})")),
    ("medication", re.compile(r"(?:用药|当前用药)\s*[:：]\s*(?P<value>[^。；;\n]{1,60})")),
    (
        "measurement",
        re.compile(
            r"(?P<label>血糖|血压|体温|尿酸|胆固醇|心率)\s*[:：]\s*"
            r"(?P<value>[0-9./]+(?:\s*[A-Za-z%/]+)?)"
        ),
    ),
)


def extract_medical_fact_candidates(
    text: str, *, source_type: str
) -> list[MedicalFactCandidate]:
    """Extract conservative candidates from an allowed source."""

    if source_type not in {"user_message", "personal_document"}:
        return []
    patterns = _USER_PATTERNS if source_type == "user_message" else _DOCUMENT_PATTERNS
    candidates: list[MedicalFactCandidate] = []
    seen: set[tuple[str, str]] = set()
    for raw_sentence in _SENTENCE_SPLIT.split(text):
        sentence = " ".join(raw_sentence.split()).strip()
        if not sentence:
            continue
        if _NEGATION.search(sentence) or _HISTORICAL.search(sentence):
            continue
        if source_type == "user_message" and (
            _THIRD_PERSON.search(sentence) or _QUESTION.search(sentence)
        ):
            continue
        for memory_type, pattern in patterns:
            for match in pattern.finditer(sentence):
                value = match.group("value").strip(" ，,。；;")
                if not value or value in {"无", "否认", "不详", "未知"}:
                    continue
                label = (match.groupdict().get("label") or memory_type).strip()
                key = (memory_type, f"{label}:{value}")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    MedicalFactCandidate(
                        memory_type=memory_type,
                        content=_candidate_content(memory_type, label, value),
                        structured_value={"name": label, "value": value},
                        confidence=0.98 if source_type == "personal_document" else 0.75,
                    )
                )
    return candidates


def build_session_summary(messages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build an auditable extractive summary without inventing medical facts."""

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = " ".join(str(message.get("content") or "").split()).strip()
        if role in {"user", "assistant"} and content:
            normalized.append(
                {
                    "turn_id": str(message.get("turn_id") or ""),
                    "role": role,
                    "content": content,
                }
            )
    digest_input = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    lines = [
        f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
        for item in normalized
    ]
    content = "\n".join(lines)
    if len(content) > 6000:
        content = "【摘要仅保留最近消息】\n" + content[-5970:]
    return {
        "content": content or "本次会话无可摘要消息。",
        "structured_summary": {
            "turn_ids": list(dict.fromkeys(item["turn_id"] for item in normalized)),
            "user_message_count": sum(item["role"] == "user" for item in normalized),
            "assistant_message_count": sum(
                item["role"] == "assistant" for item in normalized
            ),
            "extractive": True,
        },
        "source_digest": digest,
        "message_count": len(normalized),
    }


def candidate_key(
    *, source_type: str, source_id: str, memory_type: str, content: str
) -> str:
    raw = "\x1f".join((source_type, source_id, memory_type, content))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _candidate_content(memory_type: str, label: str, value: str) -> str:
    prefixes = {
        "allergy": "过敏",
        "condition": "疾病",
        "medication": "用药",
        "measurement": label,
    }
    return f"{prefixes[memory_type]}：{value}"
