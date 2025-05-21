from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Request, HTTPException, Depends

from core import User
from core.database.orm_query import orm_get_coin
from backend.app.configuration import Server, get_db, CoinResponse

import logging

router = APIRouter(tags=["Main"])

logger = logging.getLogger("app_fastapi.main")

@router.get("/")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@router.get("/team_page")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "team.html",
        {"request": request}
    )

@router.get("/contact_page")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "contact.html",
        {"request": request}
    )

@router.get("/faq")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "faq.html",
        {"request": request}
    )

@router.get("/profile_page")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "faq.html",
        {"request": request}
    )

@router.post("/users/")
async def create_user(
    username: str,
    email: str,
    session: AsyncSession = Depends(get_db)
):
    # Проверка уникальности email
    result = await session.execute(select(User).where(User.email == email))

    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    new_user = User(username=username, email=email)
    session.add(new_user)
    await session.commit()
    return new_user

@router.put("/users/{user_id}")
async def update_username(
    user_id: int,
    new_username: str,
    session: AsyncSession = Depends(get_db)
):
    result = await session.execute(
        update(User)
        .where(User.id == user_id)
        .values(username=new_username)
        .returning(User)
    )
    updated_user = result.scalars().first()
    
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await session.commit()
    return updated_user

@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    session: AsyncSession = Depends(get_db)
):
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@router.get("/coin/{coin_name}", response_model=CoinResponse)
async def get_coin(
    coin_name: str,
    session: AsyncSession = Depends(get_db)
):
    coin = await orm_get_coin(session, coin_name)
    
    if not coin:
        raise HTTPException(status_code=404, detail="Coin not found")
    
    return coin