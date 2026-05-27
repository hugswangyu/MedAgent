"""Pydantic 请求/响应模型。"""

from pydantic import BaseModel
from typing import Optional, Literal


# --- Auth ---
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class UserResponse(BaseModel):
    username: str
    is_admin: bool = False


class MessageResponse(BaseModel):
    message: str


# --- Chat ---
class ChatRequest(BaseModel):
    message: str
    session_id: str
    knowledge_base: str = "全科"


# --- Sessions ---
class SessionSummary(BaseModel):
    session_id: str
    message_count: int
    updated_at: str


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionMessage(BaseModel):
    type: Literal["human", "ai"]
    content: str
    rag_trace: Optional[dict] = None


class SessionDetailResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]


# --- Documents ---
class JobStepItem(BaseModel):
    key: str
    label: str
    percent: int
    status: Literal["pending", "running", "completed", "failed"]
    message: str = ""


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    message: str = ""
    steps: list[JobStepItem] = []


class UploadResponse(BaseModel):
    job_id: str
    message: str


class DeleteResponse(BaseModel):
    job_id: str
    message: str


class DocumentItem(BaseModel):
    filename: str
    file_type: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
