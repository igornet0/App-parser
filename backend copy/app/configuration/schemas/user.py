from typing import Optional
from pydantic import BaseModel

class UserCreateResponse(BaseModel):
    login: str
    email: Optional[str] = None
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    login: str
    balance: float

class TokenData(BaseModel):
    login: Optional[str] = None

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
