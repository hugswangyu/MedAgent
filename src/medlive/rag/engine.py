from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from functools import partial
from math import ceil
from pathlib import Path
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from medlive.rag.settings import Settings

from .schemas import SUPPORTED_MODES, ConversationOptions, QueryOptions

FOLLOWUP_PHRASES = {
    "接着说",
    "继续",
    "继续说",
    "详细说",
    "详细说说",
    "展开说说",
    "展开讲讲",
    "然后呢",
    "还有呢",
    "再说说",
    "再讲讲",
    "具体点",
    "讲详细点",
    "说详细点",
}

CODE_LIKE_ENTITY_RE = re.compile(
    r"(^src/|^chunk-|^doc-|https?://|[/\\].+\.(ts|tsx|js|jsx|py|md|json|ya?ml|toml|sh|go|rs|java|kt|swift|rb|php|c|cpp|h|hpp)$)",
    re.IGNORECASE,
)


def _is_followup_query(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= 12 and any(p in stripped for p in FOLLOWUP_PHRASES)


def rewrite_query(query: str, conversation: ConversationOptions) -> tuple[str, bool]:
    if not conversation.rewrite_followup or not _is_followup_query(query):
        return query, False
    anchor = (conversation.last_query or "").strip()
    if not anchor:
        return query, False
    rewritten = f"上一轮问题：{anchor}\n当前追问：{query}\n请围绕上一轮主题继续补充。"
    return rewritten, True


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "value") and value.__class__.__name__.endswith("Status"):
        return value.value
    return value


def _strip_chunk_content(chunks: list[dict[str, Any]], include_content: bool) -> list[dict[str, Any]]:
    if include_content:
        return chunks
    return [{k: v for k, v in chunk.items() if k != "content"} for chunk in chunks]


def _is_topic_entity(name: str) -> bool:
    """判断一个实体名是否适合作为知识库主题展示。"""

    text = name.strip()
    if len(text) < 2 or len(text) > 80:
        return False
    if CODE_LIKE_ENTITY_RE.search(text):
        return False
    if text.count("/") or text.count("\\"):
        return False
    if sum(ch in "*`{}[]<>" for ch in text) >= 2:
        return False
    return sum(ch.isalpha() for ch in text) != 0


def _build_topic_preview(candidates: list[str]) -> str:
    """把一组实体名压缩成文档主题预览。"""

    filtered = [item for item in candidates if _is_topic_entity(item)]
    picked = filtered[:3] if filtered else candidates[:3]
    return " / ".join(picked)


class RagEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rag: LightRAG | None = None
        self._write_lock = asyncio.Lock()
        self._background_jobs: set[asyncio.Task[None]] = set()

    async def initialize(self) -> None:
        Path(self.settings.absolute_user_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.settings.absolute_working_dir).mkdir(parents=True, exist_ok=True)
        Path(self.settings.absolute_upload_dir).mkdir(parents=True, exist_ok=True)

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            keyword_extraction: bool = False,
            **kwargs: Any,
        ) -> str:
            return await openai_complete_if_cache(
                self.settings.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                keyword_extraction=keyword_extraction,
                api_key=self.settings.llm_api_key or None,
                base_url=self.settings.llm_base_url or None,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=self.settings.embedding_dim,
            max_token_size=self.settings.max_embed_tokens,
            func=partial(
                openai_embed.func,
                model=self.settings.embedding_model,
                base_url=self.settings.embedding_base_url or None,
                api_key=self.settings.embedding_api_key or None,
            ),
        )

        self.rag = LightRAG(
            working_dir=self.settings.absolute_working_dir,
            workspace=self.settings.workspace,
            kv_storage=self.settings.kv_storage,
            vector_storage=self.settings.vector_storage,
            graph_storage=self.settings.graph_storage,
            doc_status_storage=self.settings.doc_status_storage,
            llm_model_func=llm_model_func,
            llm_model_name=self.settings.llm_model,
            embedding_func=embedding_func,
            chunk_token_size=self.settings.chunk_token_size,
            chunk_overlap_token_size=self.settings.chunk_overlap_token_size,
            embedding_batch_num=self.settings.embedding_batch_num,
            embedding_func_max_async=self.settings.embedding_func_max_async,
            llm_model_max_async=self.settings.llm_model_max_async,
            max_parallel_insert=self.settings.max_parallel_insert,
            entity_extract_max_gleaning=self.settings.entity_extract_max_gleaning,
            enable_llm_cache=self.settings.enable_llm_cache,
            enable_llm_cache_for_entity_extract=self.settings.enable_llm_cache_for_entity_extract,
        )
        await self.rag.initialize_storages()

    async def finalize(self) -> None:
        for task in list(self._background_jobs):
            task.cancel()
        if self._background_jobs:
            await asyncio.gather(*self._background_jobs, return_exceptions=True)
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None

    def ensure_ready(self) -> LightRAG:
        if self.rag is None:
            raise RuntimeError("LightRAG engine is not initialized")
        return self.rag

    def defaults(self) -> dict[str, Any]:
        return {
            "supported_modes": SUPPORTED_MODES,
            "profiles": {
                "default": self._profile_defaults("default").model_dump(),
                "voice": self._profile_defaults("voice").model_dump(),
            },
            "storage": {
                "user_data_dir": self.settings.absolute_user_data_dir,
                "working_dir": self.settings.absolute_working_dir,
                "upload_dir": self.settings.absolute_upload_dir,
                "workspace": self.settings.workspace,
                "kb_id": self.settings.kb_id,
                "kb_name": self.settings.kb_name,
                "kv_storage": self.settings.kv_storage,
                "vector_storage": self.settings.vector_storage,
                "graph_storage": self.settings.graph_storage,
                "doc_status_storage": self.settings.doc_status_storage,
            },
        }

    def ready_state(self) -> dict[str, Any]:
        return {
            "initialized": self.rag is not None,
            "provider_configured": self.settings.provider_ready(),
            "llm_model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_dim": self.settings.embedding_dim,
            "working_dir": self.settings.absolute_working_dir,
            "user_data_dir": self.settings.absolute_user_data_dir,
            "upload_dir": self.settings.absolute_upload_dir,
            "workspace": self.settings.workspace,
            "kb_id": self.settings.kb_id,
            "kb_name": self.settings.kb_name,
        }

    def _profile_defaults(self, profile: str) -> QueryOptions:
        if profile == "voice":
            return QueryOptions(
                mode=self.settings.voice_mode,  # type: ignore[arg-type]
                top_k=self.settings.voice_top_k,
                chunk_top_k=self.settings.voice_chunk_top_k,
                enable_rerank=self.settings.voice_enable_rerank,
                include_references=False,
                include_chunk_content=False,
                context_max_chars=self.settings.voice_context_max_chars,
            )
        return QueryOptions(
            mode=self.settings.default_mode,  # type: ignore[arg-type]
            top_k=self.settings.top_k,
            chunk_top_k=self.settings.chunk_top_k,
            enable_rerank=self.settings.enable_rerank,
            include_references=True,
            include_chunk_content=False,
        )

    def resolve_options(self, profile: str, options: QueryOptions) -> QueryOptions:
        base = self._profile_defaults(profile).model_dump()
        incoming = options.model_dump(exclude_none=True)
        base.update(incoming)
        return QueryOptions(**base)

    def build_query_param(self, options: QueryOptions, *, only_need_context: bool) -> QueryParam:
        values: dict[str, Any] = {
            "mode": options.mode,
            "only_need_context": only_need_context,
            "only_need_prompt": False,
            "stream": False,
            "hl_keywords": options.hl_keywords,
            "ll_keywords": options.ll_keywords,
            "enable_rerank": options.enable_rerank,
            "include_references": options.include_references,
        }
        for field_name in (
            "top_k",
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
            "response_type",
        ):
            value = getattr(options, field_name)
            if value is not None:
                values[field_name] = value
        return QueryParam(**{k: v for k, v in values.items() if v is not None})

    async def query_context(
        self,
        query: str,
        profile: str,
        options: QueryOptions,
        conversation: ConversationOptions,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rag = self.ensure_ready()
        resolved = self.resolve_options(profile, options)
        effective_query, rewritten = rewrite_query(query, conversation)
        started = time.perf_counter()

        if len(effective_query.strip()) < 3:
            metrics = self._query_metrics(started, resolved, cache_hit=False)
            return {
                "hit": False,
                "query": query,
                "effective_query": effective_query,
                "rewritten": rewritten,
                "context": "",
                "context_truncated": False,
                "references": [],
                "chunks": [],
            }, metrics | {"chunks_count": 0}

        if resolved.mode == "bypass":
            metrics = self._query_metrics(started, resolved, cache_hit=False)
            return {
                "hit": False,
                "query": query,
                "effective_query": effective_query,
                "rewritten": rewritten,
                "context": "",
                "context_truncated": False,
                "references": [],
                "chunks": [],
            }, metrics | {"chunks_count": 0}

        param = self.build_query_param(resolved, only_need_context=True)
        # only_need_context=True 时，LightRAG 只做检索并返回上下文和原始 chunks。
        # 这里不会执行答案生成 LLM 调用。
        result = await rag.aquery_llm(effective_query, param=param)
        payload = self._extract_context_payload(result, resolved)
        payload.update({"query": query, "effective_query": effective_query, "rewritten": rewritten})
        metrics = self._query_metrics(started, resolved, cache_hit=False)
        metrics["chunks_count"] = len(result.get("data", {}).get("chunks", []) or [])
        return payload, metrics

    async def query_data(
        self,
        query: str,
        profile: str,
        options: QueryOptions,
        conversation: ConversationOptions,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rag = self.ensure_ready()
        resolved = self.resolve_options(profile, options)
        effective_query, rewritten = rewrite_query(query, conversation)
        started = time.perf_counter()
        param = self.build_query_param(resolved, only_need_context=True)
        result = await rag.aquery_data(effective_query, param=param)
        metrics = self._query_metrics(started, resolved, cache_hit=False)
        metrics["chunks_count"] = len(result.get("data", {}).get("chunks", []) or [])
        return {
            "query": query,
            "effective_query": effective_query,
            "rewritten": rewritten,
            "result": _to_jsonable(result),
        }, metrics

    async def query_answer(
        self,
        query: str,
        profile: str,
        options: QueryOptions,
        conversation: ConversationOptions,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rag = self.ensure_ready()
        resolved = self.resolve_options(profile, options)
        effective_query, rewritten = rewrite_query(query, conversation)
        started = time.perf_counter()
        param = self.build_query_param(resolved, only_need_context=False)
        result = await rag.aquery_llm(effective_query, param=param)
        llm_response = result.get("llm_response", {}) or {}
        metrics = self._query_metrics(started, resolved, cache_hit=False)
        metrics["chunks_count"] = len(result.get("data", {}).get("chunks", []) or [])
        return {
            "query": query,
            "effective_query": effective_query,
            "rewritten": rewritten,
            "answer": llm_response.get("content") or "",
            "result": _to_jsonable({k: v for k, v in result.items() if k != "llm_response"}),
        }, metrics

    async def enqueue_documents(
        self,
        *,
        texts: list[str],
        file_sources: list[str],
        document_ids: list[str],
        track_id: str,
    ) -> dict[str, Any]:
        """把已解析文本入队给 LightRAG。"""

        rag = self.ensure_ready()
        if len(texts) != len(file_sources) or len(texts) != len(document_ids):
            raise ValueError("texts, file_sources and document_ids length must match")
        async with self._write_lock:
            await rag.apipeline_enqueue_documents(
                texts,
                ids=document_ids,
                file_paths=file_sources,
                track_id=track_id,
            )
        self._schedule_background_pipeline()
        return {
            "track_id": track_id,
            "processing_mode": "async",
            "count": len(texts),
            "kb_id": self.settings.kb_id,
            "kb_name": self.settings.kb_name,
        }

    async def documents(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        rag = self.ensure_ready()
        docs, total = await rag.doc_status.get_docs_paginated(page=page, page_size=page_size)
        counts = await rag.doc_status.get_all_status_counts()
        total_pages = ceil(total / page_size) if total else 0
        return {
            "documents": [
                self._with_kb({"document_id": doc_id, **_to_jsonable(status)}) for doc_id, status in docs
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1 and total_pages > 0,
            "status_counts": counts,
        }

    async def document_detail(self, document_id: str) -> dict[str, Any]:
        """读取单个文档状态、原文和文本块。"""

        rag = self.ensure_ready()
        status = await rag.doc_status.get_by_id(document_id)
        if not status:
            raise KeyError(f"document not found: {document_id}")

        full_doc = await rag.full_docs.get_by_id(document_id)
        json_status = _to_jsonable(status)
        chunks_list = json_status.get("chunks_list") or []
        chunks = []
        if chunks_list:
            raw_chunks = await rag.text_chunks.get_by_ids(chunks_list)
            chunks = _to_jsonable(raw_chunks)

        return self._with_kb({
            "document_id": document_id,
            "status": json_status,
            "content": (full_doc or {}).get("content", ""),
            "file_path": json_status.get("file_path") or (full_doc or {}).get("file_path"),
            "chunks": chunks,
            "chunks_count": len(chunks),
        })

    async def delete_document(
        self,
        document_id: str,
        *,
        delete_llm_cache: bool = False,
    ) -> dict[str, Any]:
        """删除单个文档及其派生文本块、实体、关系和向量数据。"""

        rag = self.ensure_ready()
        async with self._write_lock:
            result = await rag.adelete_by_doc_id(
                document_id,
                delete_llm_cache=delete_llm_cache,
            )
        return _to_jsonable(result)

    async def clear_documents(self) -> dict[str, Any]:
        """清空当前 RAG workspace 的全部索引和上传源文件。"""

        rag = self.ensure_ready()
        storages = [
            rag.text_chunks,
            rag.full_docs,
            rag.full_entities,
            rag.full_relations,
            rag.entity_chunks,
            rag.relation_chunks,
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.chunks_vdb,
            rag.chunk_entity_relation_graph,
            rag.doc_status,
        ]

        async with self._write_lock:
            results = []
            for storage in storages:
                if storage is None:
                    continue
                storage_name = storage.__class__.__name__
                namespace = getattr(storage, "namespace", "")
                result = await storage.drop()
                results.append(
                    {
                        "storage": storage_name,
                        "namespace": namespace,
                        "result": _to_jsonable(result),
                    }
                )

            deleted_files = 0
            file_errors: list[dict[str, str]] = []
            upload_dir = Path(self.settings.absolute_upload_dir)
            if upload_dir.exists():
                for path in upload_dir.iterdir():
                    if not path.is_file():
                        continue
                    try:
                        path.unlink()
                        deleted_files += 1
                    except OSError as exc:
                        file_errors.append({"path": str(path), "error": str(exc)})

        failed_storages = [
            item
            for item in results
            if item["result"].get("status") not in {"success", "ok"}
        ]
        return {
            "status": "partial_success" if failed_storages or file_errors else "success",
            "storage_results": results,
            "failed_storage_count": len(failed_storages),
            "deleted_source_files": deleted_files,
            "file_errors": file_errors,
        }

    async def job(self, job_id: str) -> dict[str, Any]:
        rag = self.ensure_ready()
        docs = await rag.aget_docs_by_track_id(job_id)
        return {
            "job_id": job_id,
            "documents": [
                self._with_kb({"document_id": doc_id, **_to_jsonable(status)}) for doc_id, status in docs.items()
            ],
            "total": len(docs),
            "kb_id": self.settings.kb_id,
            "kb_name": self.settings.kb_name,
        }

    async def knowledge_overview(
        self,
        *,
        entity_limit: int = 20,
        relation_limit: int = 12,
        document_limit: int = 10,
        topic_limit: int = 8,
    ) -> dict[str, Any]:
        """返回知识库的实体、关系和文档主题概览。"""

        rag = self.ensure_ready()
        status_counts = await rag.doc_status.get_all_status_counts()
        total_documents = int(status_counts.get("all", 0) or 0)
        docs, _ = await rag.doc_status.get_docs_paginated(
            page=1,
            page_size=max(total_documents or 0, 1),
        )
        if total_documents == 0:
            total_documents = len(docs)
        documents = [(doc_id, _to_jsonable(status)) for doc_id, status in docs]
        processed = [(doc_id, status) for doc_id, status in documents if status.get("status") == "processed"]
        processed_ids = [doc_id for doc_id, _ in processed]

        if not processed_ids:
            return {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "summary": {
                    "total_documents": total_documents,
                    "processed_documents": status_counts.get("processed", 0),
                    "failed_documents": status_counts.get("failed", 0),
                    "pending_documents": total_documents
                    - status_counts.get("processed", 0)
                    - status_counts.get("failed", 0),
                    "total_chunks": 0,
                    "total_entities": 0,
                    "total_relationships": 0,
                },
                "topics": [],
                "top_entities": [],
                "top_relationships": [],
                "documents": [
                    {
                        "document_id": doc_id,
                        "kb_id": self.settings.kb_id,
                        "kb_name": self.settings.kb_name,
                        "file_path": status.get("file_path"),
                        "status": status.get("status"),
                        "chunks_count": status.get("chunks_count", 0),
                        "updated_at": status.get("updated_at") or status.get("created_at"),
                        "topic_preview": "",
                        "top_entities": [],
                    }
                    for doc_id, status in documents[:document_limit]
                ],
            }

        full_entities_raw = await rag.full_entities.get_by_ids(processed_ids)
        full_relations_raw = await rag.full_relations.get_by_ids(processed_ids)
        full_entities = [_to_jsonable(item or {}) for item in full_entities_raw]
        full_relations = [_to_jsonable(item or {}) for item in full_relations_raw]

        entity_mentions = Counter[str]()
        entity_documents = Counter[str]()
        relation_mentions = Counter[tuple[str, str]]()
        relation_documents = Counter[tuple[str, str]]()
        doc_entities: dict[str, list[str]] = {}

        for (doc_id, _status), entity_payload in zip(processed, full_entities, strict=False):
            unique_entities: list[str] = []
            seen_entities: set[str] = set()
            for raw_name in entity_payload.get("entity_names") or []:
                name = str(raw_name).strip()
                if not name:
                    continue
                entity_mentions[name] += 1
                if name in seen_entities:
                    continue
                seen_entities.add(name)
                entity_documents[name] += 1
                unique_entities.append(name)
            doc_entities[doc_id] = unique_entities

        for relation_payload in full_relations:
            seen_pairs: set[tuple[str, str]] = set()
            for raw_pair in relation_payload.get("relation_pairs") or []:
                if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                    continue
                source = str(raw_pair[0]).strip()
                target = str(raw_pair[1]).strip()
                if not source or not target:
                    continue
                pair = (source, target)
                relation_mentions[pair] += 1
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                relation_documents[pair] += 1

        unique_entities = list(entity_documents.keys())
        entity_chunk_payloads = await rag.entity_chunks.get_by_ids(unique_entities) if unique_entities else []
        entity_chunk_map = {
            name: _to_jsonable(payload or {})
            for name, payload in zip(unique_entities, entity_chunk_payloads, strict=False)
        }

        ranked_entities = sorted(
            unique_entities,
            key=lambda name: (
                entity_documents[name],
                entity_chunk_map.get(name, {}).get("count", 0),
                entity_mentions[name],
                name.lower(),
            ),
            reverse=True,
        )

        top_entities: list[dict[str, Any]] = []
        for name in ranked_entities[:entity_limit]:
            try:
                degree = await rag.chunk_entity_relation_graph.node_degree(name)
            except Exception:
                degree = 0
            top_entities.append(
                {
                    "name": name,
                    "mention_count": entity_mentions[name],
                    "document_count": entity_documents[name],
                    "chunk_count": entity_chunk_map.get(name, {}).get("count", 0),
                    "degree": degree,
                    "is_topic_like": _is_topic_entity(name),
                }
            )

        ranked_topics = [
            name
            for name in ranked_entities
            if _is_topic_entity(name)
        ]
        topics = [
            {
                "name": name,
                "document_count": entity_documents[name],
                "chunk_count": entity_chunk_map.get(name, {}).get("count", 0),
                "mention_count": entity_mentions[name],
            }
            for name in ranked_topics[:topic_limit]
        ]

        ranked_relations = sorted(
            relation_documents.keys(),
            key=lambda pair: (
                relation_documents[pair],
                relation_mentions[pair],
                pair[0].lower(),
                pair[1].lower(),
            ),
            reverse=True,
        )
        top_relationships = [
            {
                "source": source,
                "target": target,
                "document_count": relation_documents[(source, target)],
                "mention_count": relation_mentions[(source, target)],
            }
            for source, target in ranked_relations[:relation_limit]
        ]

        document_items: list[dict[str, Any]] = []
        for doc_id, status in documents[:document_limit]:
            entities = doc_entities.get(doc_id, [])
            ranked_doc_entities = sorted(
                entities,
                key=lambda name: (
                    entity_documents[name],
                    entity_chunk_map.get(name, {}).get("count", 0),
                    entity_mentions[name],
                    name.lower(),
                ),
                reverse=True,
            )
            topic_preview = _build_topic_preview(ranked_doc_entities)
            document_items.append(
                {
                    "document_id": doc_id,
                    "kb_id": self.settings.kb_id,
                    "kb_name": self.settings.kb_name,
                    "file_path": status.get("file_path"),
                    "status": status.get("status"),
                    "chunks_count": status.get("chunks_count", 0),
                    "updated_at": status.get("updated_at") or status.get("created_at"),
                    "topic_preview": topic_preview,
                    "top_entities": ranked_doc_entities[:5],
                }
            )

        pending_documents = total_documents - status_counts.get("processed", 0) - status_counts.get("failed", 0)
        total_chunks = sum((status.get("chunks_count") or 0) for _, status in processed)

        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {
                "total_documents": total_documents,
                "processed_documents": status_counts.get("processed", 0),
                "failed_documents": status_counts.get("failed", 0),
                "pending_documents": pending_documents,
                "total_chunks": total_chunks,
                "total_entities": len(unique_entities),
                "total_relationships": len(relation_documents),
            },
            "topics": topics,
            "top_entities": top_entities,
            "top_relationships": top_relationships,
            "documents": document_items,
        }

    def _schedule_background_pipeline(self) -> None:
        task = asyncio.create_task(self._run_background_pipeline())
        self._background_jobs.add(task)
        task.add_done_callback(self._background_jobs.discard)

    async def _run_background_pipeline(self) -> None:
        rag = self.ensure_ready()
        async with self._write_lock:
            await rag.apipeline_process_enqueue_documents()

    def _extract_context_payload(self, result: dict[str, Any], options: QueryOptions) -> dict[str, Any]:
        data = result.get("data", {}) or {}
        llm_response = result.get("llm_response", {}) or {}
        context = str(llm_response.get("content") or "").strip()
        if self._is_empty_context_text(context):
            context = ""
        limit = options.context_max_chars
        truncated = False
        if limit and len(context) > limit:
            context = context[:limit].strip()
            truncated = True
        chunks = data.get("chunks", []) or []
        references = data.get("references", []) or []
        return {
            "hit": bool(context) and result.get("status") == "success",
            "context": context,
            "context_truncated": truncated,
            "references": self._with_kb_list(_to_jsonable(references)) if options.include_references else [],
            "chunks": self._with_kb_list(_to_jsonable(
                _strip_chunk_content(chunks, options.include_chunk_content)
            ))
            if options.include_references
            else [],
        }

    def _query_metrics(self, started: float, options: QueryOptions, *, cache_hit: bool) -> dict[str, Any]:
        return {
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "mode": options.mode,
            "top_k": options.top_k,
            "chunk_top_k": options.chunk_top_k,
            "enable_rerank": options.enable_rerank,
            "cache_hit": cache_hit,
            "kb_id": self.settings.kb_id,
            "kb_name": self.settings.kb_name,
        }

    def _with_kb(self, item: dict[str, Any]) -> dict[str, Any]:
        """给返回对象追加知识库来源。"""

        return {**item, "kb_id": self.settings.kb_id, "kb_name": self.settings.kb_name}

    def _with_kb_list(self, items: Any) -> list[Any]:
        """给列表里的字典项追加知识库来源。"""

        if not isinstance(items, list):
            return []
        return [self._with_kb(item) if isinstance(item, dict) else item for item in items]

    @staticmethod
    def _is_empty_context_text(context: str) -> bool:
        """识别 LightRAG 在无上下文时返回的占位文本。"""

        normalized = context.strip().lower()
        return not normalized or "[no-context]" in normalized
