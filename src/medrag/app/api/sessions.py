"""会话管理端点：GET /sessions, GET /sessions/{id}, DELETE /sessions/{id}。"""

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_current_user
from ..schemas import MessageResponse, SessionListResponse, SessionDetailResponse
from ..session_store import get_sessions, get_session, delete_session

router = APIRouter()


@router.get("", response_model=SessionListResponse)
async def list_sessions(_current_user=Depends(get_current_user)):
    sessions = get_sessions()
    return SessionListResponse(sessions=sessions)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def load_session(session_id: str, _current_user=Depends(get_current_user)):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return session


@router.delete("/{session_id}", response_model=MessageResponse)
async def remove_session(session_id: str, _current_user=Depends(get_current_user)):
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return MessageResponse(message=f"已删除会话 {session_id}")
