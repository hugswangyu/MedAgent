"""Authenticated Phase 3 controlled-memory API."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from medrag.app.auth_manager import AuthUser
from medrag.app.dependencies import get_current_user
from medrag.infrastructure.storage import phase1_repository

router = APIRouter()


class MemoryCorrection(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    structured_value: dict[str, Any] = Field(default_factory=dict)
    memory_type: str | None = Field(default=None, min_length=1, max_length=100)
    confidence: float = Field(default=1.0, ge=0, le=1)


@router.get("")
async def list_memories(
    memory_status: list[
        Literal["proposed", "confirmed", "superseded", "rejected"]
    ]
    | None = Query(default=None, alias="status"),
    current_user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    items = phase1_repository.list_medical_fact_memories(
        user_id=current_user.user_id,
        statuses=memory_status,
    )
    return {"memories": items, "total": len(items)}


@router.get("/export")
async def export_memories(
    response: Response,
    current_user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    response.headers["Content-Disposition"] = (
        'attachment; filename="medagent-memory-export.json"'
    )
    return phase1_repository.export_controlled_memory(user_id=current_user.user_id)


@router.post("/{memory_id}/confirm")
async def confirm_memory(
    memory_id: str, current_user: AuthUser = Depends(get_current_user)
) -> dict[str, Any]:
    return _transition(current_user.user_id, memory_id, "confirmed")


@router.post("/{memory_id}/reject")
async def reject_memory(
    memory_id: str, current_user: AuthUser = Depends(get_current_user)
) -> dict[str, Any]:
    return _transition(current_user.user_id, memory_id, "rejected")


@router.post("/{memory_id}/correct")
async def correct_memory(
    memory_id: str,
    payload: MemoryCorrection,
    current_user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        item = phase1_repository.correct_medical_fact_memory(
            user_id=current_user.user_id,
            memory_id=memory_id,
            content=payload.content.strip(),
            structured_value=payload.structured_value,
            memory_type=payload.memory_type,
            confidence=payload.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return item


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str, current_user: AuthUser = Depends(get_current_user)
) -> Response:
    if not phase1_repository.delete_medical_fact_memory(
        user_id=current_user.user_id, memory_id=memory_id
    ):
        raise HTTPException(status_code=404, detail="memory not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _transition(user_id: str, memory_id: str, target: str) -> dict[str, Any]:
    item = phase1_repository.set_medical_fact_status(
        user_id=user_id, memory_id=memory_id, status=target
    )
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="memory not found or transition is not allowed",
        )
    return item
