"""Phase 1 PostgreSQL identity and ownership control plane."""

from __future__ import annotations

import hmac
import os
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from psycopg2 import IntegrityError

from medrag.app.auth_manager import (
    AuthUser,
    create_worker_bootstrap_token,
    create_worker_token,
    decode_worker_bootstrap_token,
    decode_worker_token,
)
from medrag.app.dependencies import get_current_user
from medrag.infrastructure.storage import phase1_repository

router = APIRouter()
bootstrap_bearer = HTTPBearer(auto_error=False)
CONTROL_PLANE_KEY_HEADER = Header(default=None)


def require_control_plane_key(
    x_control_plane_key: str | None = CONTROL_PLANE_KEY_HEADER,
) -> None:
    expected = os.getenv("MEDAGENT_CONTROL_PLANE_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="control plane key is not configured")
    if not x_control_plane_key or not hmac.compare_digest(x_control_plane_key, expected):
        raise HTTPException(status_code=401, detail="invalid control plane key")


CONTROL_PLANE_AUTH = Depends(require_control_plane_key)


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _created_lease_seconds() -> int:
    return _positive_env_int("MEDAGENT_VOICE_SESSION_CREATED_LEASE_SECONDS", 120)


def _active_lease_seconds() -> int:
    return _positive_env_int("MEDAGENT_VOICE_SESSION_LEASE_SECONDS", 120)


class KnowledgeBaseRegistration(BaseModel):
    kb_id: str


class KnowledgeBaseStatus(BaseModel):
    status: Literal["active", "deleted"]


class VoiceSessionCreate(BaseModel):
    session_id: str
    knowledge_base_id: str
    room_name: str
    client_id: str | None = None
    client_type: Literal["android", "web", "test"] = "web"
    participant_identity: str | None = None
    token_expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VoiceSessionClaim(BaseModel):
    session_id: str
    binding_version: int
    room_name: str
    livekit_job_id: str


def _worker_claims(
    credentials: HTTPAuthorizationCredentials | None,
) -> dict[str, Any]:
    claims = decode_worker_token(credentials.credentials) if credentials else None
    if claims is None:
        raise HTTPException(status_code=401, detail="invalid worker token")
    if not phase1_repository.validate_claimed_worker_binding(
        session_id=str(claims["sid"]),
        user_id=str(claims["sub"]),
        knowledge_base_id=str(claims["kid"]),
        livekit_job_id=str(claims["job"]),
    ):
        raise HTTPException(status_code=409, detail="worker binding is no longer active")
    return claims


@router.get("/me")
async def me(current_user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "is_admin": current_user.is_admin,
        "status": current_user.status,
        "token_version": current_user.token_version,
    }


@router.post("/knowledge-bases")
async def register_knowledge_base(
    payload: KnowledgeBaseRegistration,
    current_user: AuthUser = Depends(get_current_user),
    _service: None = CONTROL_PLANE_AUTH,
) -> dict[str, Any]:
    try:
        return phase1_repository.register_knowledge_base(
            kb_id=payload.kb_id, owner_user_id=current_user.user_id
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/knowledge-bases")
async def list_knowledge_bases(
    current_user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    items = phase1_repository.list_owned_knowledge_bases(current_user.user_id)
    return {"knowledge_bases": items, "total": len(items)}


@router.get("/knowledge-bases/{kb_id}")
async def get_knowledge_base(
    kb_id: str, current_user: AuthUser = Depends(get_current_user)
) -> dict[str, Any]:
    item = phase1_repository.get_owned_knowledge_base(kb_id, current_user.user_id)
    if item is None:
        raise HTTPException(status_code=404, detail="knowledge base not found")
    return item


@router.put("/knowledge-bases/{kb_id}/status")
async def update_knowledge_base_status(
    kb_id: str,
    payload: KnowledgeBaseStatus,
    current_user: AuthUser = Depends(get_current_user),
    _service: None = CONTROL_PLANE_AUTH,
) -> dict[str, str]:
    if not phase1_repository.set_knowledge_base_status(
        kb_id=kb_id, owner_user_id=current_user.user_id, status=payload.status
    ):
        raise HTTPException(status_code=404, detail="knowledge base not found")
    return {"status": payload.status}


@router.post("/voice-sessions")
async def create_voice_session(
    payload: VoiceSessionCreate,
    current_user: AuthUser = Depends(get_current_user),
    _service: None = CONTROL_PLANE_AUTH,
) -> dict[str, Any]:
    if phase1_repository.get_owned_knowledge_base(
        payload.knowledge_base_id, current_user.user_id
    ) is None:
        raise HTTPException(status_code=404, detail="knowledge base not found")
    try:
        binding = phase1_repository.create_voice_session_binding(
            user_id=current_user.user_id,
            session_id=payload.session_id,
            knowledge_base_id=payload.knowledge_base_id,
            room_name=payload.room_name,
            client_id=payload.client_id,
            client_type=payload.client_type,
            participant_identity=payload.participant_identity,
            token_expires_at=payload.token_expires_at,
            metadata=payload.metadata,
            lease_seconds=_created_lease_seconds(),
        )
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="voice session already exists or user has an open session") from exc
    binding["worker_bootstrap_token"] = create_worker_bootstrap_token(
        session_id=payload.session_id,
        binding_version=int(binding["binding_version"]),
    )
    return binding


@router.get("/voice-sessions/{session_id}")
async def get_voice_session(
    session_id: str, current_user: AuthUser = Depends(get_current_user)
) -> dict[str, Any]:
    binding = phase1_repository.get_voice_session_binding(session_id, current_user.user_id)
    if binding is None:
        raise HTTPException(status_code=404, detail="voice session not found")
    return binding


@router.post("/voice-sessions/{session_id}/end")
async def end_voice_session(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
    _service: None = CONTROL_PLANE_AUTH,
) -> dict[str, str]:
    if not phase1_repository.end_voice_session_binding(
        session_id=session_id, user_id=current_user.user_id
    ):
        raise HTTPException(status_code=404, detail="voice session not found")
    return {"status": "ended"}


@router.post("/worker/voice-sessions/claim")
async def claim_voice_session(
    payload: VoiceSessionClaim,
    credentials: HTTPAuthorizationCredentials | None = Depends(bootstrap_bearer),
) -> dict[str, Any]:
    claims = decode_worker_bootstrap_token(credentials.credentials) if credentials else None
    if claims is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid worker bootstrap token")
    if str(claims["sid"]) != payload.session_id or int(claims["ver"]) != payload.binding_version:
        raise HTTPException(status_code=409, detail="voice session binding version mismatch")
    binding = phase1_repository.claim_voice_session_binding(
        session_id=payload.session_id,
        expected_version=payload.binding_version,
        room_name=payload.room_name,
        livekit_job_id=payload.livekit_job_id,
        lease_seconds=_active_lease_seconds(),
    )
    if binding is None:
        raise HTTPException(status_code=409, detail="voice session already claimed or binding mismatch")
    binding["worker_token"] = create_worker_token(
        user_id=binding["user_id"],
        session_id=binding["session_id"],
        knowledge_base_id=binding["knowledge_base_id"],
        livekit_job_id=binding["livekit_job_id"],
    )
    return binding


@router.post("/worker/token")
async def refresh_worker_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bootstrap_bearer),
) -> dict[str, str]:
    claims = _worker_claims(credentials)
    if not phase1_repository.renew_voice_session_lease(
        session_id=str(claims["sid"]),
        user_id=str(claims["sub"]),
        knowledge_base_id=str(claims["kid"]),
        livekit_job_id=str(claims["job"]),
        lease_seconds=_active_lease_seconds(),
    ):
        raise HTTPException(status_code=409, detail="worker lease cannot be renewed")
    return {
        "worker_token": create_worker_token(
            user_id=str(claims["sub"]),
            session_id=str(claims["sid"]),
            knowledge_base_id=str(claims["kid"]),
            livekit_job_id=str(claims["job"]),
        )
    }


@router.post("/worker/voice-sessions/end")
async def worker_end_voice_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(bootstrap_bearer),
) -> dict[str, str]:
    claims = _worker_claims(credentials)
    phase1_repository.end_voice_session_binding(
        session_id=str(claims["sid"]), user_id=str(claims["sub"])
    )
    return {"status": "ended"}


@router.post("/internal/voice-sessions/cleanup")
async def cleanup_stale_voice_sessions(
    _service: None = CONTROL_PLANE_AUTH,
) -> dict[str, int]:
    return {"expired": phase1_repository.expire_stale_voice_sessions()}
