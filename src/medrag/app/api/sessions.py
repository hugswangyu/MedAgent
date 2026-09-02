"""会话管理端点：GET /sessions, GET /sessions/{id}, DELETE /sessions/{id}。"""

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_current_user
from ..schemas import MessageResponse, SessionListResponse, SessionDetailResponse
from ..session_store import (
    delete_session,
    finalize_session,
    get_session,
    get_sessions,
)

router = APIRouter()


@router.get("", response_model=SessionListResponse)
async def list_sessions(current_user=Depends(get_current_user)):
    sessions = get_sessions(username=current_user.username)
    return SessionListResponse(sessions=sessions)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def load_session(session_id: str, current_user=Depends(get_current_user)):
    session = get_session(session_id, username=current_user.username)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return session


@router.post("/{session_id}/finalize")
async def finalize_text_session(
    session_id: str,
    summary_version: int = 1,
    current_user=Depends(get_current_user),
):
    if summary_version < 1:
        raise HTTPException(status_code=422, detail="summary_version 必须大于 0")
    try:
        return finalize_session(
            session_id,
            user_id=current_user.user_id,
            summary_version=summary_version,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在"
        ) from exc


@router.delete("/{session_id}", response_model=MessageResponse)
async def remove_session(session_id: str, current_user=Depends(get_current_user)):
    try:
        finalize_session(session_id, user_id=current_user.user_id)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在"
        ) from exc
    ok = delete_session(session_id, username=current_user.username)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return MessageResponse(message=f"已删除会话 {session_id}")
