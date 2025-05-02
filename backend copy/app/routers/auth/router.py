
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from core import User, settings
from core.database.orm_query import orm_get_user, orm_add_user
from backend.app.configuration import (Server, get_db, 
                                       get_password_hash,
                                       UserResponse, UserCreateResponse,
                                       Token, verify_password, create_refresh_token,
                                       get_current_user, create_access_token)

import logging

router = APIRouter(prefix="/auth", tags=["auth"])

logger = logging.getLogger("app_fastapi.auth")

@router.get("/")
async def read_root(request: Request):
    pass

@router.post("/login/", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_db)):
    user = await orm_get_user(session, form_data)

    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/register/", response_model=UserResponse)
async def register(user: UserCreateResponse, session: AsyncSession = Depends(get_db)):

    db_user = await orm_get_user(session, user)

    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)

    await orm_add_user(session, login=user.login,
                              hashed_password=hashed_password,
                              email=user.email)
    
    db_user = await orm_get_user(session, user)

    return UserResponse(db_user)

@router.post("/refresh-token/", response_model=Token)
async def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, settings.security.refresh_secret_key, algorithms=[settings.security.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    new_access_token = create_access_token(data={"sub": username})
    new_refresh_token = create_refresh_token(data={"sub": username})
    
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }