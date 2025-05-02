# файл для query запросов

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from random import randint 

from core.database.models import (User, Coin, Timeseries, Portfolio, Transaction,
                                  News, NewsModel, RiskModel, Agent, MMM,)
from backend.app.configuration import UserResponse
##################### Добавляем юзера в БД #####################################

async def orm_add_user(
        session: AsyncSession,
        login: str,
        hashed_password: str,
        email: str | None = None,
        user_telegram_id: int | None = None
) -> User:
    
    query = select(User).where(User.login == login)
    result = await session.execute(query)

    if result.first() is None:
        session.add(
            User(login=login,
                 password=hashed_password,
                 user_telegram_id=user_telegram_id,
                 email=email)
        )
        await session.commit()

    else:
        return result.first()
    
async def orm_get_user(session: AsyncSession, response: UserResponse):
    if response.email:
        query = select(User).where(User.email == response.email)
    elif response.login:
        query = select(User).where(User.login == response.login)

    result = await session.execute(query)

    return result.first()

async def verify_user(session: AsyncSession, login: str, hashed_password: str) -> User | None:
    query = select(User).where(User.login == login)
    result = await session.execute(query)
    user = result.scalar()

    if user and user.password == hashed_password:
        return user
    
    return None

async def orm_update_user_place(session: AsyncSession, user_id: int, place_id: int):
    query = update(User).where(User.user_id == user_id).values(place=place_id)
    await session.execute(query)
    await session.commit()

async def orm_update_user_phone(session: AsyncSession, user_id: int, phone: str):
    query = update(User).where(User.user_id == user_id).values(phone=phone)
    await session.execute(query)
    await session.commit()


##################### Добавляем монеты в БД #####################################

async def orm_add_coin(
        session: AsyncSession,
        name: str,
        price_now: float = 0
) -> Coin:
    
    query = select(Coin).where(Coin.name == name)
    result = await session.execute(query)

    if not result.scalars().first():
        session.add(
            Coin(name=name,
                 price_now=price_now)
        )
        await session.commit()
        return await orm_get_coin(session, name)

async def orm_get_coin(session: AsyncSession, name: str) -> Coin:
    query = select(Coin).where(Coin.name == name)
    result = await session.execute(query)
    return result.scalar()