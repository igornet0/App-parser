from enum import Enum as PyEnum
import uuid
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr
from datetime import datetime

class OrderStatus(str, PyEnum):

    OPEN = 'open'
    APPROVE = 'approve'
    CANCELED = 'cancel'

class OrderType(str, PyEnum):
    BUY = 'buy'
    SELL = 'sell'

class OrderCreate(BaseModel):
    type: OrderType
    coin_id: int
    amount: float
    price: float

class OrderCancel(BaseModel):
    id: int

class OrderUpdateAmount(BaseModel):
    id: int
    amount: float

class OrderResponse(BaseModel):
    id: int
    type: OrderType
    coin_id: int
    amount_orig: float
    amount: float
    price: float
    status: OrderStatus
    created: datetime

class PortfolioResponse(BaseModel):
    coin_id: int
    amount: float