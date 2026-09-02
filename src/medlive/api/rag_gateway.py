"""管理 API 访问内部 RAG Core Service 的统一网关。"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

import aiohttp
from fastapi import UploadFile

from medlive.config.settings import AppSettings, load_rag_client_settings
from medlive.rag.service import wait_for_rag_ready


@dataclass(slots=True)
class GatewayResponse:
    """统一描述管理 API 返回给前端的结果。"""

    status_code: int
    body: dict[str, Any]


@dataclass(slots=True)
class GatewayFileResponse:
    """统一描述内部 RAG 文件响应。"""

    status_code: int
    body: bytes
    headers: dict[str, str]
    error_body: dict[str, Any] | None = None


def envelope(
    *,
    request_id: str | None = None,
    data: dict[str, Any] | list[Any] | None = None,
    metrics: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    """生成统一的管理 API 响应体。"""

    return {
        "request_id": request_id or str(uuid.uuid4()),
        "status": status,
        "data": data,
        "metrics": metrics or {},
        "error": error,
    }


class RagGateway:
    """把前端管理 API 请求转发到内部 RAG Core Service。"""

    def __init__(self, settings: AppSettings) -> None:
        """绑定应用级配置。"""

        self.settings = settings

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> GatewayResponse:
        """转发 GET 请求。"""

        return await self._request("GET", path, params=params)

    async def get_documents(self, path: str, *, params: dict[str, Any] | None = None) -> GatewayResponse:
        """转发文档列表请求并归一化字段。"""

        result = await self.get(path, params=params)
        return self._map_data(result, self._normalize_documents_payload)

    async def get_document_detail(self, path: str) -> GatewayResponse:
        """转发文档详情请求并归一化字段。"""

        result = await self.get(path)
        return self._map_data(result, self._normalize_document_detail)

    async def get_file(self, path: str, *, params: dict[str, Any] | None = None) -> GatewayFileResponse:
        """转发原文件请求，保留文件响应头。"""

        fallback_request_id = str(uuid.uuid4())
        ready_state = await asyncio.to_thread(
            wait_for_rag_ready,
            timeout_ms=self.settings.api.rag_ready_timeout_ms,
        )
        if not ready_state.ready:
            return GatewayFileResponse(
                status_code=503,
                body=b"",
                headers={},
                error_body=envelope(
                    request_id=fallback_request_id,
                    status="error",
                    error={"type": "RagServiceNotReady", "message": ready_state.error or "RAG 服务未就绪"},
                    metrics={"rag_service_status": ready_state.status},
                ),
            )

        rag_settings = load_rag_client_settings(self.settings.user_data_dir)
        timeout = aiohttp.ClientTimeout(total=max(self.settings.api.rag_gateway_timeout_ms, 100) / 1000.0)
        target_url = self._target_url(rag_settings.base_url, path)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(
                    target_url,
                    params=self._query_params(params),
                    headers=self._headers(rag_settings.api_key, has_json=False, has_form=False),
                ) as response,
            ):
                content_type = response.headers.get("content-type", "")
                if response.status >= 400 or "application/json" in content_type:
                    payload = await self._read_payload(response)
                    normalized = self._normalize_payload(
                        payload,
                        status_code=response.status,
                        fallback_request_id=fallback_request_id,
                    )
                    return GatewayFileResponse(
                        status_code=normalized.status_code,
                        body=b"",
                        headers={},
                        error_body=normalized.body,
                    )
                body = await response.read()
                headers = self._file_headers(response)
                return GatewayFileResponse(status_code=response.status, body=body, headers=headers)
        except Exception as exc:
            return GatewayFileResponse(
                status_code=502,
                body=b"",
                headers={},
                error_body=envelope(
                    request_id=fallback_request_id,
                    status="error",
                    error={"type": type(exc).__name__, "message": str(exc)},
                ),
            )

    async def get_job(self, path: str) -> GatewayResponse:
        """转发任务查询请求并归一化文档字段。"""

        result = await self.get(path)
        return self._map_data(result, self._normalize_job_payload)

    async def post_json(self, path: str, *, payload: dict[str, Any]) -> GatewayResponse:
        """转发 JSON POST 请求。"""

        return await self._request("POST", path, json_body=payload)

    async def patch_json(self, path: str, *, payload: dict[str, Any]) -> GatewayResponse:
        """转发 JSON PATCH 请求。"""

        return await self._request("PATCH", path, json_body=payload)

    async def delete(self, path: str, *, params: dict[str, Any] | None = None) -> GatewayResponse:
        """转发 DELETE 请求。"""

        return await self._request("DELETE", path, params=params)

    async def post_files(self, path: str, *, files: list[UploadFile]) -> GatewayResponse:
        """转发 multipart 文件上传请求。"""

        form = aiohttp.FormData()
        for uploaded in files:
            raw = await uploaded.read()
            form.add_field(
                "files",
                raw,
                filename=uploaded.filename or "uploaded_file",
                content_type=uploaded.content_type or "application/octet-stream",
            )
        return await self._request("POST", path, form_data=form, upload=True)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        form_data: aiohttp.FormData | None = None,
        upload: bool = False,
    ) -> GatewayResponse:
        """执行一次统一的内部 RAG 请求。"""

        fallback_request_id = str(uuid.uuid4())
        ready_state = await asyncio.to_thread(
            wait_for_rag_ready,
            timeout_ms=self.settings.api.rag_ready_timeout_ms,
        )
        if not ready_state.ready:
            return GatewayResponse(
                status_code=503,
                body=envelope(
                    request_id=fallback_request_id,
                    status="error",
                    error={"type": "RagServiceNotReady", "message": ready_state.error or "RAG 服务未就绪"},
                    metrics={"rag_service_status": ready_state.status},
                ),
            )

        rag_settings = load_rag_client_settings(self.settings.user_data_dir)
        timeout_ms = (
            max(self.settings.api.rag_gateway_upload_timeout_ms, 30_000)
            if upload
            else max(self.settings.api.rag_gateway_timeout_ms, 100)
        )
        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
        headers = self._headers(rag_settings.api_key, has_json=bool(json_body), has_form=bool(form_data))
        target_url = self._target_url(rag_settings.base_url, path)

        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.request(
                    method,
                    target_url,
                    params=self._query_params(params),
                    json=json_body,
                    data=form_data,
                    headers=headers,
                ) as response,
            ):
                payload = await self._read_payload(response)
                return self._normalize_payload(
                    payload,
                    status_code=response.status,
                    fallback_request_id=fallback_request_id,
                )
        except Exception as exc:
            return GatewayResponse(
                status_code=502,
                body=envelope(
                    request_id=fallback_request_id,
                    status="error",
                    error={"type": type(exc).__name__, "message": str(exc)},
                ),
            )

    @staticmethod
    def _headers(api_key: str, *, has_json: bool, has_form: bool) -> dict[str, str]:
        """构造转发请求头。"""

        headers: dict[str, str] = {}
        if has_json:
            headers["Content-Type"] = "application/json"
        if has_form:
            headers.pop("Content-Type", None)
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    @staticmethod
    def _query_params(params: dict[str, Any] | None) -> dict[str, str | int | float] | None:
        """Normalize query parameters for aiohttp/yarl.

        yarl intentionally rejects Python bool values even though FastAPI query
        parameters often arrive as bools. Convert them before forwarding.
        """

        if not params:
            return None

        normalized: dict[str, str | int | float] = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                normalized[key] = "true" if value else "false"
            elif isinstance(value, str | int | float):
                normalized[key] = value
            else:
                normalized[key] = str(value)

        return normalized or None

    @staticmethod
    def _target_url(base_url: str, path: str) -> str:
        """拼接内部 RAG 服务地址。"""

        normalized = path if path.startswith("/") else f"/{path}"
        return f"{base_url.rstrip('/')}{normalized}"

    @staticmethod
    def _file_headers(response: aiohttp.ClientResponse) -> dict[str, str]:
        """保留前端预览原文件需要的安全响应头。"""

        allowed = {
            "content-type",
            "content-disposition",
            "etag",
            "last-modified",
            "cache-control",
        }
        return {
            key: value
            for key, value in response.headers.items()
            if key.lower() in allowed
        }

    @staticmethod
    async def _read_payload(response: aiohttp.ClientResponse) -> Any:
        """按内容类型读取上游响应。"""

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return await response.json()
        return await response.text()

    @staticmethod
    def _normalize_payload(payload: Any, *, status_code: int, fallback_request_id: str) -> GatewayResponse:
        """把上游响应归一成统一 envelope。"""

        if isinstance(payload, dict) and "status" in payload and "request_id" in payload:
            body = {
                "request_id": payload.get("request_id") or fallback_request_id,
                "status": payload.get("status") or ("error" if payload.get("error") else "ok"),
                "data": payload.get("data"),
                "metrics": payload.get("metrics") or {},
                "error": payload.get("error"),
            }
            return GatewayResponse(status_code=status_code, body=body)

        if status_code >= 400:
            message = RagGateway._error_message(payload)
            return GatewayResponse(
                status_code=status_code,
                body=envelope(
                    request_id=fallback_request_id,
                    status="error",
                    error={"type": "UpstreamError", "message": message},
                ),
            )

        return GatewayResponse(
            status_code=status_code,
            body=envelope(request_id=fallback_request_id, data=payload),
        )

    @staticmethod
    def _error_message(payload: Any) -> str:
        """把各种上游错误体压成一条可读信息。"""

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail
            if isinstance(detail, dict):
                message = detail.get("message") or detail.get("detail")
                if message:
                    return str(message)
            if payload.get("error"):
                return str(payload["error"])
            return json.dumps(payload, ensure_ascii=False)
        return str(payload)

    @classmethod
    def _map_data(cls, result: GatewayResponse, mapper: Any) -> GatewayResponse:
        """只转换成功 envelope 中的 data 字段。"""

        if result.body.get("status") != "ok":
            return result
        body = dict(result.body)
        body["data"] = mapper(body.get("data"))
        return GatewayResponse(status_code=result.status_code, body=body)

    @classmethod
    def _normalize_documents_payload(cls, data: Any) -> dict[str, Any]:
        """归一化文档列表响应。"""

        if not isinstance(data, dict):
            return {"documents": [], "total": 0, "raw": data}
        payload = dict(data)
        documents = data.get("documents")
        payload["documents"] = [
            cls._normalize_document_summary(item) for item in documents if isinstance(item, dict)
        ] if isinstance(documents, list) else []
        return payload

    @classmethod
    def _normalize_job_payload(cls, data: Any) -> dict[str, Any]:
        """归一化任务查询响应。"""

        if not isinstance(data, dict):
            return {"job_id": "", "documents": [], "total": 0, "raw": data}
        payload = dict(data)
        documents = data.get("documents")
        payload["documents"] = [
            cls._normalize_document_summary(item) for item in documents if isinstance(item, dict)
        ] if isinstance(documents, list) else []
        return payload

    @classmethod
    def _normalize_document_detail(cls, data: Any) -> dict[str, Any]:
        """归一化文档详情响应。"""

        if not isinstance(data, dict):
            return {"document_id": "", "status": "unknown", "content": "", "chunks": [], "raw": data}
        status_payload = data.get("status") if isinstance(data.get("status"), dict) else {}
        summary_source = {**status_payload, **data}
        summary = cls._normalize_document_summary(summary_source)
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            chunks = []
        return {
            **summary,
            "content": data.get("content") or "",
            "chunks": chunks,
            "chunks_count": data.get("chunks_count") or summary.get("chunks_count") or len(chunks),
            "status_raw": status_payload,
            "raw": data,
        }

    @staticmethod
    def _normalize_document_summary(item: dict[str, Any]) -> dict[str, Any]:
        """归一化单个文档摘要字段。"""

        status_value = item.get("status")
        if isinstance(status_value, dict):
            status_raw = status_value
            status = status_raw.get("status") or "unknown"
        else:
            status_raw = item
            status = status_value or item.get("doc_status") or "unknown"
        chunks = item.get("chunks")
        chunks_count = item.get("chunks_count")
        if chunks_count is None:
            chunks_count = item.get("chunk_count")
        if chunks_count is None:
            chunks_list = item.get("chunks_list")
            chunks_count = len(chunks_list) if isinstance(chunks_list, list) else 0
        return {
            "document_id": item.get("document_id") or item.get("doc_id") or item.get("id") or "",
            "kb_id": item.get("kb_id") or "",
            "kb_name": item.get("kb_name") or "",
            "original_filename": item.get("original_filename") or "",
            "file_path": item.get("file_path") or item.get("file_source") or item.get("source") or "",
            "source_file_path": item.get("source_file_path") or "",
            "source_file_exists": bool(item.get("source_file_exists")),
            "source_file_size": item.get("source_file_size") or 0,
            "source_sha256": item.get("source_sha256") or "",
            "content_type": item.get("content_type") or "",
            "extension": item.get("extension") or "",
            "parse_status": item.get("parse_status") or "",
            "index_status": item.get("index_status") or "",
            "status": status,
            "chunks_count": chunks_count,
            "content": item.get("content") or "",
            "content_summary": item.get("content_summary") or item.get("summary") or "",
            "content_length": item.get("content_length") or item.get("content_len") or 0,
            "chunks": chunks if isinstance(chunks, list) else [],
            "error_msg": item.get("error_msg") or item.get("error") or item.get("message"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "track_id": item.get("track_id"),
            "status_raw": status_raw,
            "raw": item,
        }
