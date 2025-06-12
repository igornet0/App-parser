from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, String, Float, Enum, ForeignKey, select, update
from sqlalchemy.orm import relationship, selectinload
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from pydantic import BaseModel
from enum import Enum as PyEnum
from contextlib import asynccontextmanager
from typing import Optional

# Настройки базы данных
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/exchange_db"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Базовый класс моделей
Base = declarative_base()

class OrderStatus(str, PyEnum):
    OPEN = 'open'
    FILLED = 'filled'
    CANCELED = 'canceled'

class OrderType(str, PyEnum):
    BUY = 'buy'
    SELL = 'sell'

class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    balance = Column(Float, default=10000.0)  # Стартовый баланс
    portfolio = relationship("Portfolio", back_populates="user")
    orders = relationship("Order", back_populates="user")

class Portfolio(Base):
    __tablename__ = 'portfolio'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    asset = Column(String, index=True)
    quantity = Column(Float)
    user = relationship("User", back_populates="portfolio")

class Order(Base):
    __tablename__ = 'orders'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    type = Column(Enum(OrderType))
    asset = Column(String, index=True)
    quantity = Column(Float)
    price = Column(Float)
    status = Column(Enum(OrderStatus), default=OrderStatus.OPEN)
    user = relationship("User", back_populates="orders")

# Pydantic схемы
class UserCreate(BaseModel):
    balance: float = 10000.0

class OrderCreate(BaseModel):
    type: OrderType
    asset: str
    quantity: float
    price: float

class OrderResponse(BaseModel):
    id: uuid.UUID
    type: OrderType
    asset: str
    quantity: float
    price: float
    status: OrderStatus

class PortfolioResponse(BaseModel):
    asset: str
    quantity: float

# Инициализация приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(lifespan=lifespan)

# Dependency для получения сессии БД
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

@app.post("/users/", response_model=UserCreate)
async def create_user(db: AsyncSession = Depends(get_db)):
    user = User()
    db.add(user)
    await db.flush()
    return user

@app.get("/users/{user_id}/balance")
async def get_balance(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"balance": user.balance}

@app.get("/users/{user_id}/portfolio")
async def get_portfolio(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id, options=[selectinload(User.portfolio)])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return [{"asset": item.asset, "quantity": item.quantity} for item in user.portfolio]

@app.post("/orders/", response_model=OrderResponse)
async def create_order(order_data: OrderCreate, user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        # Блокируем пользователя для изменения
        user = await db.get(User, user_id, with_for_update=True)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Проверяем достаточно ли средств/активов
        if order_data.type == OrderType.BUY:
            total_cost = order_data.quantity * order_data.price
            if user.balance < total_cost:
                raise HTTPException(status_code=400, detail="Insufficient funds")
            # Блокируем средства
            user.balance -= total_cost
        else:  # OrderType.SELL
            # Ищем актив в портфеле
            portfolio_item = next(
                (item for item in user.portfolio if item.asset == order_data.asset), 
                None
            )
            if not portfolio_item or portfolio_item.quantity < order_data.quantity:
                raise HTTPException(status_code=400, detail="Insufficient assets")
            # Блокируем активы
            portfolio_item.quantity -= order_data.quantity
        
        # Создаем заявку
        order = Order(
            user_id=user_id,
            **order_data.dict()
        )
        db.add(order)
        await db.flush()
        return order

@app.post("/orders/{order_id}/execute")
async def execute_order(order_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        # Блокируем заявку и пользователя
        order = await db.get(Order, order_id, with_for_update=True)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        if order.status != OrderStatus.OPEN:
            raise HTTPException(status_code=400, detail="Order is not open")
        
        user = await db.get(User, order.user_id, with_for_update=True, options=[selectinload(User.portfolio)])
        
        # Исполняем заявку
        if order.type == OrderType.BUY:
            # Ищем актив в портфеле
            portfolio_item = next(
                (item for item in user.portfolio if item.asset == order.asset), 
                None
            )
            if not portfolio_item:
                # Создаем новый актив
                portfolio_item = Portfolio(
                    user_id=user.id,
                    asset=order.asset,
                    quantity=0
                )
                db.add(portfolio_item)
                user.portfolio.append(portfolio_item)
            
            # Зачисляем актив
            portfolio_item.quantity += order.quantity
        else:  # OrderType.SELL
            # Зачисляем деньги
            user.balance += order.quantity * order.price
        
        # Обновляем статус заявки
        order.status = OrderStatus.FILLED
        return {"status": "executed"}

@app.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        order = await db.get(Order, order_id, with_for_update=True)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        if order.status != OrderStatus.OPEN:
            raise HTTPException(status_code=400, detail="Order is not open")
        
        user = await db.get(User, order.user_id, with_for_update=True)
        
        # Возвращаем средства/активы
        if order.type == OrderType.BUY:
            user.balance += order.quantity * order.price
        else:  # OrderType.SELL
            # Ищем актив в портфеле
            portfolio_item = next(
                (item for item in user.portfolio if item.asset == order.asset), 
                None
            )
            if portfolio_item:
                portfolio_item.quantity += order.quantity
        
        order.status = OrderStatus.CANCELED
        return {"status": "canceled"}