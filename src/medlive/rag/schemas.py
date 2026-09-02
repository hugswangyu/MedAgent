from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
ProfileName = Literal["default", "voice"]

SUPPORTED_MODES: list[str] = ["local", "global", "hybrid", "naive", "mix", "bypass"]


class QueryOptions(BaseModel):
    mode: QueryMode | None = None
    top_k: int | None = Field(default=None, ge=1)
    chunk_top_k: int | None = Field(default=None, ge=1)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    enable_rerank: bool | None = None
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)
    include_references: bool = False
    include_chunk_content: bool = False
    context_max_chars: int | None = Field(default=None, ge=1)
    response_type: str | None = None


class ConversationOptions(BaseModel):
    last_query: str | None = None
    rewrite_followup: bool = True


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    profile: ProfileName = "default"
    options: QueryOptions | None = None
    conversation: ConversationOptions | None = None

    # 扁平字段是前端调试查询接口的正式请求形状。
    mode: QueryMode | None = None
    top_k: int | None = Field(default=None, ge=1)
    chunk_top_k: int | None = Field(default=None, ge=1)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    enable_rerank: bool | None = None
    hl_keywords: list[str] | None = None
    ll_keywords: list[str] | None = None
    include_references: bool | None = None
    include_chunk_content: bool | None = None
    context_max_chars: int | None = Field(default=None, ge=1)
    response_type: str | None = None
    last_query: str | None = None
    rewrite_followup: bool | None = None

    @field_validator("query", mode="after")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return value.strip()

    def merged_options(self) -> QueryOptions:
        merged = self.options.model_dump() if self.options else {}
        for field_name in QueryOptions.model_fields:
            value = getattr(self, field_name, None)
            if value is not None:
                merged[field_name] = value
        return QueryOptions(**merged)

    def merged_conversation(self) -> ConversationOptions:
        merged = self.conversation.model_dump() if self.conversation else {}
        if self.last_query is not None:
            merged["last_query"] = self.last_query
        if self.rewrite_followup is not None:
            merged["rewrite_followup"] = self.rewrite_followup
        return ConversationOptions(**merged)


class TextDocumentRequest(BaseModel):
    text: str = Field(min_length=1)
    file_source: str | None = None
    document_id: str | None = None


class Envelope(BaseModel):
    request_id: str
    status: Literal["ok", "error"]
    data: dict[str, Any] | list[Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
