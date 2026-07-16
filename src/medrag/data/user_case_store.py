"""用户个人病例索引与本地检索。

该模块保存脱敏摘要和文本分块，并按 username 强制隔离。它不依赖
Milvus，因此即使向量库不可用，个人病例上下文仍能进入 RAG 链路。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from medrag.config.settings import settings
from medrag.infrastructure.storage import JsonStore


_case_store = JsonStore(str(settings.user_cases_index_path))


def _read_cases() -> list[dict]:
    data = _case_store.read()
    return data if isinstance(data, list) else []


def _write_cases(cases: list[dict]) -> None:
    _case_store.write(cases)


def add_user_case(
    username: str,
    filename: str,
    chunks: list[str],
    summary: str = "",
    document_id: Optional[str] = None,
    status: str = "ready",
) -> str:
    """新增或替换用户病例记录，返回 document_id。"""
    doc_id = document_id or str(uuid.uuid4())
    case = {
        "username": username,
        "document_id": doc_id,
        "filename": filename,
        "summary": summary,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "status": status,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    def add(cases: dict | list) -> None:
        if not isinstance(cases, list):
            raise TypeError("user case index must contain a JSON list")
        cases[:] = [
            c for c in cases
            if not (c.get("username") == username and c.get("filename") == filename)
        ]
        cases.append(case)

    _case_store.update(add, default_factory=list)
    return doc_id


def remove_user_case(username: str, filename: str) -> bool:
    def remove(cases: dict | list) -> bool:
        if not isinstance(cases, list):
            raise TypeError("user case index must contain a JSON list")
        original_count = len(cases)
        cases[:] = [
            c for c in cases
            if not (c.get("username") == username and c.get("filename") == filename)
        ]
        return len(cases) != original_count

    return _case_store.update(remove, default_factory=list)


def get_user_cases(username: str) -> list[dict]:
    return [c for c in _read_cases() if c.get("username") == username]


def get_combined_case_summary(username: str, max_chars: int = 2000) -> str:
    """合并当前用户所有病例摘要，控制 prompt 长度。"""
    parts: list[str] = []
    for case in get_user_cases(username):
        summary = (case.get("summary") or "").strip()
        if not summary:
            continue
        parts.append(f"【{case.get('filename', '病例')}】\n{summary}")
    text = "\n\n".join(parts).strip()
    return text[:max_chars]


class UserCaseRetriever:
    """基于本地病例 chunk 的轻量检索器。"""

    def search(self, query: str, username: str | None, top_k: int = 5) -> List[Dict]:
        if not username:
            return []
        keywords = _extract_keywords(query)
        scored: list[dict] = []
        for case in get_user_cases(username):
            for idx, chunk in enumerate(case.get("chunks") or []):
                score = _score_text(chunk, keywords)
                if score <= 0:
                    continue
                scored.append(
                    {
                        "source": "user_case",
                        "source_type": "user_case",
                        "username": username,
                        "document_id": case.get("document_id", ""),
                        "filename": case.get("filename", ""),
                        "chunk_index": idx,
                        "title": case.get("filename", ""),
                        "answer": chunk,
                        "text": chunk,
                        "score": round(score, 4),
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def _extract_keywords(query: str) -> list[str]:
    text = query.replace("?", "").replace("？", "").replace("，", "").replace("。", "")
    keywords: list[str] = []
    for token in text.split():
        if len(token) >= 2:
            keywords.append(token)
    if len(text) >= 2:
        keywords.extend(text[i:i + 2] for i in range(len(text) - 1))
    if len(text) >= 3:
        keywords.extend(text[i:i + 3] for i in range(len(text) - 2))
    seen: set[str] = set()
    return [kw for kw in keywords if kw and not (kw in seen or seen.add(kw))]


def _score_text(text: str, keywords: list[str]) -> float:
    if not text or not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in text)
    return hits / max(len(keywords), 1)
