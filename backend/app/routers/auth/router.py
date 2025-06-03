from typing import Annotated
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (HTTPBearer,
                              HTTPAuthorizationCredentials, 
                              OAuth2PasswordRequestForm)
from datetime import timedelta

from core import settings
from core.database.orm_query import orm_get_user_by_email, orm_get_user_by_login, orm_add_user

from backend.app.configuration import (Server, 
                                       get_password_hash,
                                       UserResponse, UserLoginResponse,
                                       Token, get_current_token_payload,
                                       verify_password, is_email,
                                       create_access_token,
                                       get_user_by_token_sub,
                                       validate_token_type)

import logging

http_bearer = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(http_bearer)],)

logger = logging.getLogger("app_fastapi.auth")


@router.post("/register/", response_model=Token)
async def register(user: UserLoginResponse = Body(), session: AsyncSession = Depends(Server.get_db)):

    db_user = await orm_get_user_by_email(session, user)

    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = await orm_get_user_by_login(session, user)

    if db_user:
        raise HTTPException(status_code=400, detail="Login already registered")
    
    hashed_password = get_password_hash(user.password)

    await orm_add_user(session, login=user.login,
                              hashed_password=hashed_password,
                              email=user.email)
    
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)

    return {
        "access_token": create_access_token(payload={"sub": user.login, "email": user.email}, 
                                            expires_delta=access_token_expires),
        "token_type": "bearer",
        "message": "User registered successfully"
    }


@router.post("/login_user/", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(Server.get_db)):
    
    unauthed_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    identifier_type = "email" if is_email(form_data.username) else "login"

    if identifier_type == "email":
        user = await orm_get_user_by_email(session, UserLoginResponse(email=form_data.username, password=form_data.password))
    else:
        user = await orm_get_user_by_login(session, UserLoginResponse(login=form_data.username, password=form_data.password))

    if not user:
        raise unauthed_exc
    
    if not verify_password(
        plain_password=form_data.password,
        hashed_password=user.password,
    ):
        raise unauthed_exc
    
    if not user.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user inactive",
        )
    
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)

    access_token = create_access_token(
        payload={"sub": user.login, "email": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, 
            "token_type": "bearer",
            "message": "User logged in successfully",}


@router.post("/refresh-token/", response_model=Token)
async def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, settings.security.refresh_secret_key, algorithms=[settings.security.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    new_access_token = create_access_token(payload={"sub": username})
    # new_refresh_token = create_refresh_token(data={"sub": username})
    
    return {
        "access_token": new_access_token,
        # "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.get("/user/me/", response_model=UserResponse)
async def auth_user_check_self_info(
    token: str = Depends(Server.oauth2_scheme),
    session: AsyncSession = Depends(Server.get_db)
):
    payload = get_current_token_payload(token)

    validate_token_type(payload, "access")
        
    user = await get_user_by_token_sub(payload, session)
    
    if user.active:
        return user
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="inactive user",
    )