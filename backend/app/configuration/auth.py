from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone

from core.settings import settings
from core.database import User
from backend.app.configuration import get_db, TokenData

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.security.secret_key, algorithm=settings.security.algorithm)

def create_refresh_token(data: dict):
    expire = datetime.now(timezone.utc) + timedelta(days=settings.security.refresh_token_expire_days)
    data.update({"exp": expire})
    return jwt.encode(data, settings.security.refresh_secret_key, algorithm=settings.security.algorithm)

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.security.refresh_secret_key, algorithms=[settings.security.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.login == token_data.login).first()
    if user is None:
        raise credentials_exception
    return user