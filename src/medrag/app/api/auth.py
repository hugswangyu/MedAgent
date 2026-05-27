"""认证端点：POST /auth/login, POST /auth/register, GET /auth/me。"""

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth_manager import (
    create_access_token,
    create_user,
    get_user,
    verify_user,
)
from ..dependencies import get_current_user
from ..schemas import LoginRequest, TokenResponse, UserResponse

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    username = body.username.strip()
    password = body.password.strip()
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名和密码不能为空")

    user = verify_user(username, password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    token = create_access_token(username)
    return TokenResponse(access_token=token, username=username)


@router.post("/register", response_model=TokenResponse)
def register(body: LoginRequest):
    username = body.username.strip()
    password = body.password.strip()
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名和密码不能为空")

    if get_user(username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已存在")

    user = create_user(username, password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="注册失败")

    token = create_access_token(username)
    return TokenResponse(access_token=token, username=username)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    return UserResponse(username=current_user.username, is_admin=current_user.is_admin)
