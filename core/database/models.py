# модели для БД
from sqlalchemy import DateTime, ForeignKey, Float, String, BigInteger, func, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from passlib.context import CryptContext

from core.database.base import Base

class User(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(50), nullable=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    user_telegram_id: Mapped[int] = mapped_column(BigInteger)
    balance: Mapped[float] = mapped_column(Float, default=0)


class Coin(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    price_now: Mapped[float] = mapped_column(Float, default=0)

    timeseries: Mapped[list['Timeseries']] = relationship(back_populates='coin')
    portfolio: Mapped[list['Portfolio']] = relationship(back_populates='coin')
    transaction: Mapped[list['Transaction']] = relationship(back_populates='coin')

class Portfolio(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id', ondelete='CASCADE'), nullable=False)
    coin: Mapped['Coin'] = relationship(back_populates='portfolio')
    user: Mapped['User'] = relationship(backref='portfolio')
    amount: Mapped[float] = mapped_column(Float, default=0)


class Timeseries(Base):

    __tablename__ = 'timeseries'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))  
    timestamp: Mapped[str] = mapped_column(String(50)) 
    path_dataset: Mapped[str] = mapped_column(String(100), unique=True)

    coin: Mapped['Coin'] = relationship(back_populates='timeseries')


class Transaction(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(30), default="new")
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id', ondelete='CASCADE'), nullable=False)

    coin: Mapped['Coin'] = relationship(back_populates='transaction')
    user: Mapped['User'] = relationship(backref='transaction')

    def set_status(self, new_status):
        assert not new_status in ["new", "open", "close", "cancel"]

        self.status = new_status


class News(Base):
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(50), default="news")
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    text: Mapped[str] = mapped_column(String(1000), nullable=False)
    date: Mapped[DateTime] = mapped_column(DateTime, default=func.now())


class Agent(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path_model: Mapped[str] = mapped_column(String(100), unique=True)
    version: Mapped[int] = mapped_column(Integer, unique=True)

class NewsModel(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path_model: Mapped[str] = mapped_column(String(100), unique=True)
    version: Mapped[int] = mapped_column(Integer, unique=True)

class RiskModel(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path_model: Mapped[str] = mapped_column(String(100), unique=True)
    version: Mapped[int] = mapped_column(Integer, unique=True)

class MMM(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path_model: Mapped[str] = mapped_column(String(100), unique=True)
    version: Mapped[int] = mapped_column(Integer, unique=True)
