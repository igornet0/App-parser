# файл для query запросов
from typing import Tuple, Dict
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from random import randint 

from core.database.models import (User, Coin, Timeseries, DataTimeseries, Portfolio, Transaction,
                                  News, NewsModel, RiskModel, Agent, MMM)

from backend.app.configuration import UserLoginResponse
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
        return result.scalars().first()
    
async def orm_get_user_by_login(session: AsyncSession, response: UserLoginResponse) -> Tuple[User, Dict[str, str]] | None:
    if not response.login:
        return None
    
    query = select(User).where(User.login == response.login)
     
    result = await session.execute(query)

    return result.scalars().first()

async def orm_get_user_by_email(session: AsyncSession, response: UserLoginResponse) -> Tuple[User, Dict[str, str]] | None:
    if not response.email:
        return None
    
    query = select(User).where(User.email == response.email)

    result = await session.execute(query)

    return result.scalars().first()

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
    
    return await orm_get_coin_by_name(session, name)

async def orm_get_coin_by_name(session: AsyncSession, name: str) -> Coin:
    query = select(Coin).where(Coin.name == name)
    result = await session.execute(query)
    return result.scalar()

async def orm_update_coin_price(session: AsyncSession, name: str, price_now: float):
    query = update(Coin).where(Coin.name == name).values(price_now=price_now)
    await session.execute(query)
    await session.commit()

async def orm_add_timeseries(session: AsyncSession, coin: Coin | str, timestamp: str, path_dataset: str):
    if isinstance(coin, str):
        coin = await orm_get_coin_by_name(session, coin)

    if not coin:
        raise ValueError(f"Coin {coin} not found")
    
    tm = await orm_get_timeseries_by_coin(session, coin, timestamp)

    if tm:
        return await orm_update_timeseries_path(session, tm.id, path_dataset)

    timeseries = Timeseries(coin_id=coin.id, 
                            timestamp=timestamp, 
                            path_dataset=path_dataset)
    session.add(timeseries)
    await session.commit()

async def orm_update_timeseries_path(session: AsyncSession, timeseries_id: int, path_dataset: str):
    query = update(Timeseries).where(Timeseries.id == timeseries_id).values(path_dataset=path_dataset)
    await session.execute(query)
    await session.commit()

async def orm_get_timeseries_by_path(session: AsyncSession, path_dataset: str):
    query = select(Timeseries).where(Timeseries.path_dataset == path_dataset)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_timeseries_by_id(session: AsyncSession, id: int):
    query = select(Timeseries).where(Timeseries.id == id)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_timeseries_by_coin(session: AsyncSession, coin: Coin | str, timestamp: str = None):
    if isinstance(coin, str):
        coin = await orm_get_coin_by_name(session, coin)

        if not coin:
            raise ValueError(f"Coin {coin} not found")
        
    if timestamp:
        query = select(Timeseries).options(joinedload(Timeseries.coin)).where(Timeseries.coin_id == coin.id, Timeseries.timestamp == timestamp)
        result = await session.execute(query)
        return result.scalars().first()
    
    query = select(Timeseries).options(joinedload(Timeseries.coin)).where(Timeseries.coin_id == coin.id)
    result = await session.execute(query)

    return result.scalars().all()

async def orm_get_data_timeseries(session: AsyncSession, timeseries_id: int):
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries_id)
    result = await session.execute(query)
    return result.scalars().all()

async def orm_get_data_timeseries_by_datetime(session: AsyncSession, timeseries_id: int, datetime: str):
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries_id, DataTimeseries.datetime == datetime)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_add_data_timeseries(session: AsyncSession, timeseries_id: int, data_timeseries: dict):
    dt = await orm_get_data_timeseries_by_datetime(session, timeseries_id, data_timeseries["datetime"])

    if dt:
        return False
    
    session.add(DataTimeseries(timeseries_id=timeseries_id, **data_timeseries))
    await session.commit()

    return True