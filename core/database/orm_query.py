# файл для query запросов
from typing import Tuple, Dict, Literal
from datetime import datetime
from sqlalchemy import select, update, delete, desc, asc, Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from core.database.models import (User, Coin, Timeseries, 
                                  DataTimeseries, Transaction, Portfolio, 
                                  News, NewsCoin, NewsHistoryCoin,
                                  Agent, AgentAction, StatisticAgent,
                                  ML_Model, ModelAction, StatisticModel,
                                  Strategy, StrategyAgent, AgentTrain,
                                  StrategyCoin, TrainCoin)

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
    
async def orm_get_user_by_login(session: AsyncSession, response) -> Tuple[User, Dict[str, str]] | None:
    if not response.login:
        return None
    
    query = select(User).where(User.login == response.login).options(joinedload(User.portfolio))
     
    result = await session.execute(query)

    return result.scalars().first()

async def orm_get_user_by_email(session: AsyncSession, response) -> Tuple[User, Dict[str, str]] | None:
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

async def orm_get_user_balance(session: AsyncSession, user_id: int) -> float:
    query = select(User).where(User.user_id == user_id)
    result = await session.execute(query)
    return result.scalars().first().balance

async def orm_remove_user_balance(session: AsyncSession, user_id: int, amount: float):
    new_balance = await orm_get_user_balance(session, user_id) - amount

    if new_balance < 0:
        raise ValueError("Balance cannot be negative")
    
    query = update(User).where(User.user_id == user_id).values(balance=new_balance)

    await session.execute(query)
    await session.commit()

async def orm_add_user_balance(session: AsyncSession, user_id: int, amount: float):
    new_balance = await orm_get_user_balance(session, user_id) + amount

    query = update(User).where(User.user_id == user_id).values(balance=new_balance)

    await session.execute(query)
    await session.commit()


##################### Добавляем Agents и Models в БД #####################################
async def orm_add_agent(session: AsyncSession, type_agent: str, path_model: str):
    agent = Agent(type_agent=type_agent, path_model=path_model)
    session.add(agent)
    await session.commit()

async def orm_get_agents(session: AsyncSession, type_agent: str = None, 
                         id_agent: int = None, version: str = None, 
                         active: bool = None, query_return: bool = False) -> list[Agent] | Select:
    query = select(Agent)

    if type_agent:
        query = query.where(Agent.type == type_agent)

    if id_agent:
        query = query.where(Agent.id == id_agent)

    if version:
        query = query.where(Agent.version == version)

    if active:
        query = query.where(Agent.active == active)

    if query_return:
        return query
    
    result = await session.execute(query)

    return result.scalars().all()

async def orm_get_agent_by_id(session: AsyncSession, id: int) -> Agent:
    query = select(Agent).where(Agent.id == id)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_agents_options(session: AsyncSession, type_agent: str = None, 
                         id_agent: int = None, version: str = None, 
                         active: bool = None, mod: Literal["actions", "strategies", "stata", "all"] = None) -> list[Agent]:
    query: Select = await orm_get_agents(session, type_agent, id_agent, version, active, query=True)
    
    if mod in ["actions", "all"]:
        query = query.options(joinedload(Agent.actions))

    if mod in ["strategies", "all"]:
        query = query.options(joinedload(Agent.strategies))

    if mod in ["stata", "all"]:
        query = query.options(joinedload(Agent.stata))

    result = await session.execute(query)

    return result.scalars().all()

async def orm_get_models(session: AsyncSession, type_model: str = None, 
                         id_model: int = None, version: str = None, 
                         query_return: bool = False) -> list[ML_Model] | Select:
    query = select(ML_Model)

    if type_model:
        query = query.where(ML_Model.type == type_model)

    if id_model:
        query = query.where(ML_Model.id == id_model)

    if version:
        query = query.where(ML_Model.version == version)

    if query_return:
        return query
    
    query = query.options(joinedload(ML_Model.actions), joinedload(ML_Model.stata))
    result = await session.execute(query)
    return result.scalars().all()

async def orm_get_models_options(session: AsyncSession, type_model: str = None, 
                         id_model: int = None, version: str = None, 
                         mod: Literal["actions", "stata", "all"] = None) -> list[Agent]:
    
    query: Select = await orm_get_models(session, type_model, id_model, version, query=True)
    
    if mod in ["actions", "all"]:
        query = query.options(joinedload(ML_Model.actions))

    if mod in ["stata", "all"]:
        query = query.options(joinedload(ML_Model.stata))

    result = await session.execute(query)

    return result.scalars().all()

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

async def orm_get_coins(session: AsyncSession, parsed: bool = None) -> list[Coin]:
    query = select(Coin) 

    if parsed:
        query = query.where(Coin.parsed == parsed)
        
    result = await session.execute(query)

    return result.scalars().all()

async def orm_get_coin_by_id(session: AsyncSession, id: int, parsed: bool = None) -> Coin:
    query = select(Coin).where(Coin.id == id)
    
    if parsed:
        query = query.where(Coin.parsed == parsed)

    query = query.options(joinedload(Coin.timeseries))

    result = await session.execute(query)
    return result.scalar()

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

async def orm_get_timeseries_by_coin(session: AsyncSession, coin: Coin | str | int, timeframe: str = None) -> list[Timeseries] | Timeseries:
    if isinstance(coin, str):
        coin = await orm_get_coin_by_name(session, coin)
    elif isinstance(coin, int):
        coin = await orm_get_coin_by_id(session, coin)
    
    if not coin:
        raise ValueError(f"Coin {coin} not found")
    
    query = select(Timeseries).options(joinedload(Timeseries.coin)).where(Timeseries.coin_id == coin.id)
        
    if timeframe:
        query = query.where(Timeseries.timestamp == timeframe)
    
    result = await session.execute(query)

    if timeframe:
        return result.scalars().first()

    return result.scalars().all()

async def orm_get_data_timeseries(session: AsyncSession, timeseries_id: int) -> list[DataTimeseries]:
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries_id)
    result = await session.execute(query)
    return result.scalars().all()

async def orm_get_data_timeseries_by_datetime(session: AsyncSession, timeseries_id: int, datetime: str) -> DataTimeseries:
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries_id, DataTimeseries.datetime == datetime)
    result = await session.execute(query)
    return result.scalars().first()

async def paginate_coin_prices(
    session: AsyncSession, 
    coin_id: int,
    timeframe: str = "5m",
    last_timestamp: datetime = None, 
    limit: int = 100,
    sort: bool = False
) -> list[DataTimeseries]:
    
    timeseries = await orm_get_timeseries_by_coin(session, coin_id, timeframe=timeframe)

    if not timeseries:
        raise ValueError(f"Timeseries - {timeframe} for coin - {coin_id} not found")

    # Базовый запрос с сортировкой по времени (новые записи сначала)
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries.id
                                         ).order_by(desc(DataTimeseries.datetime))
    
    # Фильтр для следующей страницы
    if last_timestamp is not None:
        query = query.where(DataTimeseries.datetime < last_timestamp)
    
    # Получаем 100 записей
    result = await session.execute(query.limit(limit))

    records = result.scalars().all()

    if sort:
        records = sorted(records, key=lambda x: x.datetime)

    return records

async def orm_add_data_timeseries(session: AsyncSession, timeseries_id: int, data_timeseries: dict):
    dt = await orm_get_data_timeseries_by_datetime(session, timeseries_id, data_timeseries["datetime"])

    if dt:
        return False
    
    session.add(DataTimeseries(timeseries_id=timeseries_id, **data_timeseries))
    await session.commit()

    return True


##################### Добавляем Portfolio в БД #####################################

async def orm_get_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int) -> Portfolio:
    query = select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.coin_id == coin_id).options(selectinload(Portfolio.coin))
    result = await session.execute(query)

    coin_potfolio = result.scalars().first()

    if not coin_potfolio:
        return None

    return coin_potfolio

async def orm_add_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int, amount: float):
    coin = await orm_get_coin_portfolio(session, user_id, coin_id)

    if coin:
        return await orm_update_amount_coin_portfolio(session, user_id, coin_id, coin[1] + amount)
    
    session.add(Portfolio(user_id=user_id, coin_id=coin_id, amount=amount))
    await session.commit()

async def orm_get_coins_portfolio(session: AsyncSession, user_id: int) -> Dict[Coin, float]:
    query = select(Portfolio).where(Portfolio.user_id == user_id).options(selectinload(Portfolio.coin))
    result = await session.execute(query)
    new_coins = {}
    coins_portfolio = result.scalars().all()

    for coin in coins_portfolio:
        new_coins[coin.coin] = coin.amount

    return new_coins

async def orm_delete_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int):
    query = delete(Portfolio).where(Portfolio.user_id == user_id, Portfolio.coin_id == coin_id)
    await session.execute(query)
    await session.commit()

async def orm_update_amount_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int, amount: float):
    if amount == 0:
        return await orm_delete_coin_portfolio(session, user_id, coin_id)
    
    query = update(Portfolio).where(Portfolio.user_id == user_id, Portfolio.coin_id == coin_id).values(amount=amount)
    await session.execute(query)
    await session.commit()


##################### Добавляем Transaction в БД #####################################

async def orm_add_transaction(session: AsyncSession, user_id: int, coin_id: int, type_order: Literal["buy", "sell"], amount: float, price: float):
    transaction = Transaction(user_id=user_id, 
                              coin_id=coin_id, 
                              type=type_order,
                              amount=amount, 
                              price=price)
    session.add(transaction)
    await session.commit()

async def orm_get_transactions_by_id(session: AsyncSession, transaction_id: int, status: str = None) -> Transaction:
    query = select(Transaction).where(Transaction.id == transaction_id)
    if status:
        if "!" in status:
            status = status.replace("!", "")
            query = query.where(Transaction.status != status)
        else:
            query = query.where(Transaction.status == status)

    query = query.options(selectinload(Transaction.coin))
    query = query.options(selectinload(Transaction.user))
    result = await session.execute(query)

    return result.scalars().first()


async def orm_get_user_transactions(session: AsyncSession, user_id: int, status: str = None, type_order: Literal["buy", "sell"] = None) -> list[Transaction]:
    query = select(Transaction).where(Transaction.user_id == user_id)

    if status:
        query = query.where(Transaction.status == status)

    if type_order:
        if not type_order in ["buy", "sell"]:
            raise ValueError("type_order must be 'buy' or 'sell'")
        
        query = query.where(Transaction.type == type_order)
        
    result = await session.execute(query)

    return result.scalars().all()

async def orm_get_coin_transactions(session: AsyncSession, coin_id: int, status: str = None, type_order: Literal["buy", "sell"] = None) -> list[Transaction]:
    query = select(Transaction).where(Transaction.coin_id == coin_id)

    if status:
        query = query.where(Transaction.status == status)

    if type_order:
        if not type_order in ["buy", "sell"]:
            raise ValueError("type_order must be 'buy' or 'sell'")
        
        query = query.where(Transaction.type == type_order)

    result = await session.execute(query)
    return result.scalars().all()

async def orm_get_user_coin_transactions(session: AsyncSession, user_id: int, coin_id: int, status: str = None, type_order: Literal["buy", "sell"] = None) -> Dict[Coin, Dict[str, float]]:
    query = select(Transaction).where(Transaction.user_id == user_id, Transaction.coin_id == coin_id)
    
    if status:
        query = query.where(Transaction.status == status)
    
    if type_order:
        if not type_order in ["buy", "sell"]:
            raise ValueError("type_order must be 'buy' or 'sell'")
        
        query = query.where(Transaction.type == type_order)

    query = query.options(selectinload(Transaction.coin))

    result = await session.execute(query)

    new_coins = {}
    coins_portfolio = result.scalars().all()

    for coin in coins_portfolio:
        new_coins[coin.coin] = {"id":coin.id, "amount": coin.amount, "price": coin.price}

    return new_coins

async def orm_update_transaction_status(session: AsyncSession, transaction_id: int, status: Literal["open", "approve", "close", "cancel"]):
    query = select(Transaction).where(Transaction.id == transaction_id)
    result = await session.execute(query)

    transaction = result.scalars().first()
    transaction.set_status(status)

    await session.commit()

async def orm_update_transaction_amount(session: AsyncSession, transaction_id: int, amount: float):
    if amount == 0:
        return await orm_update_transaction_status(session, transaction_id, status="approve")
    
    query = update(Transaction).where(Transaction.id == transaction_id).values(amount=amount)
    await session.execute(query)
    await session.commit()


async def orm_delete_transaction(session: AsyncSession, transaction_id: int):
    query = delete(Transaction).where(Transaction.id == transaction_id)
    await session.execute(query)
    await session.commit()