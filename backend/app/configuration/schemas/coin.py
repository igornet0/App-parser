from typing import Literal, Optional
from pydantic import BaseModel
from datetime import datetime

class TimeLineCoin(BaseModel):

    coin_id: int
    timeframe: Literal["5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M", "1y"] = "5m"
    last_timestamp: Optional[datetime] = None
    size_page: int = 100

class CoinData(BaseModel):
    datetime: datetime

    open_price: float
    close_price: float
    max_price: float
    min_price: float
    volume: float

class CoinResponse(BaseModel):
    id: int
    name: str
    price_now: float
    price_change_percentage_24h: float


class CoinResponseData(BaseModel):
    coin_id: int
    price_now: float

    coin_data: list[CoinData]
    last_timestamp: datetime = None
