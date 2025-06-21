from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm_query import (Coin, User, Transaction, Portfolio, 
                                     DataTimeseries,
                                     orm_get_coins,
                                     orm_get_coin_by_id,
                                     orm_get_data_timeseries,
                                     paginate_coin_prices,
                                     orm_get_agents,
                                     orm_get_models,
                                     orm_get_transactions_by_id,
                                     orm_get_user_transactions,
                                     orm_get_user_coin_transactions,
                                     orm_get_coin_portfolio)

from backend.Dataset import DatasetTimeseries
from backend.app.configuration import (Server, 
                                       TimeLineCoin,
                                       CoinData,
                                       CoinResponse,
                                       CoinResponseData,
                                    #    CreateStrategyResponse,
                                    #    StrategyResponse,
                                       UserResponse,
                                       OrderUpdateAmount,
                                       OrderResponse,
                                       OrderCreate,
                                       OrderCancel,
                                       OrderType,
                                       verify_authorization)

# Инициализация роутера
router = APIRouter(
    prefix="/coins",
    dependencies=[Depends(Server.http_bearer), Depends(verify_authorization)],
    tags=["Strategy"]
)

timeframe_size_k = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "4h": 24,
    "1d": 48,
    "1w": 168,
    "1M": 720,
    "1y": 8760
}

@router.get("/get_coins/", response_model=list[CoinResponse])
async def get_coins(db: AsyncSession = Depends(Server.get_db)):
    try:
        coins = await orm_get_coins(db, parsed=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get coins: {str(e)}")

    if not coins:
        raise HTTPException(status_code=404, detail="No coins found")
    
    coins = sorted(coins, key=lambda x: x.price_now, reverse=True)

    return coins 
    
    
@router.get("/get_coin/", response_model=CoinResponseData)
async def get_coin_by_id(coin_id: int = Query(..., alias="coin_id"),
                        timeframe: str = Query("5m", alias="timeframe"),
                        size_page: int = Query(100, alias="size_page"),
                        last_timestamp: Optional[str] = Query(None, alias="last_timestamp"),
                        user: User = Depends(verify_authorization),
                        db: AsyncSession = Depends(Server.get_db)):
    
    time_line_coin = TimeLineCoin(coin_id=coin_id,
                                  timeframe=timeframe,
                                  last_timestamp=datetime.strptime(last_timestamp.split("+")[0], "%Y-%m-%dT%H:%M:%S") if last_timestamp else None,
                                  size_page=size_page)
    
    try:
        coin = await orm_get_coin_by_id(db, id=time_line_coin.coin_id, 
                                        parsed=user.role != "admin")
        records = await paginate_coin_prices(
            db, 
            coin_id=time_line_coin.coin_id, 
            timeframe="5m",
            last_timestamp=time_line_coin.last_timestamp, 
            limit=time_line_coin.size_page * timeframe_size_k[timeframe],
            sort=True
        )

        if timeframe != "5m":
            records = DatasetTimeseries(records).get_time_line_loader()
            records = records.get_data()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get coin: {str(e)}")

    if not coin:
        raise HTTPException(status_code=404, detail="Coin not found")
    
    if not records:
        raise HTTPException(status_code=404, detail="No records found")
    
    coin_data = list(map(lambda x: CoinData(
            datetime=x.datetime,
            open_price=x.open,
            close_price=x.close,
            max_price=x.max,
            min_price=x.min,
            volume=x.volume
        ), records))
    
    result = CoinResponseData(
        coin_id=time_line_coin.coin_id,
        price_now=coin.price_now,
        coin_data=coin_data,
        last_timestamp=records[0].datetime
    )

    return result 
    
