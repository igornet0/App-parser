# модели для БД
from typing import Literal
from sqlalchemy import DateTime, ForeignKey, Float, String, BigInteger, func, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base


class NewsCoin(Base):
    
    news_id: Mapped[int] = mapped_column(ForeignKey('newss.id'), primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0)


class Coin(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    price_now: Mapped[float] = mapped_column(Float, default=0)
    max_price_now: Mapped[float] = mapped_column(Float, default=0)
    min_price_now: Mapped[float] = mapped_column(Float, default=0)
    open_price_now: Mapped[float] = mapped_column(Float, default=0)
    volume_now: Mapped[float] = mapped_column(Float, default=0)

    news_score_global: Mapped[Integer] = mapped_column(Integer, default=100)

    parsed: Mapped[bool] = mapped_column(Boolean, default=True)

    timeseries: Mapped[list['Timeseries']] = relationship(back_populates='coin')

    portfolio: Mapped[list['Portfolio']] = relationship(back_populates='coin')
    transaction: Mapped[list['Transaction']] = relationship(back_populates='coin')

    strategies: Mapped[list['Strategy']] = relationship(
        secondary="strategy_coins",
        back_populates='coins'
    )

    train: Mapped[list['AgentTrain']] = relationship(
        secondary="train_coins",
        back_populates='coins'
    )

    news: Mapped[list['News']] = relationship(
        secondary="news_coins",
        back_populates='news_coin'
    )


class User(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(50), unique=True, nullable=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    user_telegram_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    balance: Mapped[float] = mapped_column(Float, default=0)
    role: Mapped[str] = mapped_column(String(50), default="user")

    active: Mapped[bool] = mapped_column(Boolean, default=True)

    portfolio: Mapped[list['Portfolio']] = relationship(back_populates='user')



class Portfolio(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    price_avg: Mapped[float] = mapped_column(Float, default=0.0)
    
    coin: Mapped['Coin'] = relationship(back_populates='portfolio')
    user: Mapped['User'] = relationship(back_populates='portfolio')
    

class Timeseries(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))  
    timestamp: Mapped[str] = mapped_column(String(50)) 
    path_dataset: Mapped[str] = mapped_column(String(100), unique=True)

    coin: Mapped['Coin'] = relationship(back_populates='timeseries')


class DataTimeseries(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timeseries_id: Mapped[int] = mapped_column(ForeignKey('timeseriess.id'))  
    datetime: Mapped[DateTime] = mapped_column(DateTime, nullable=False) 
    open: Mapped[float] = mapped_column(Float)
    max: Mapped[float] = mapped_column(Float)
    min: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

class Transaction(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(30), default="open")
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))

    type: Mapped[str] = mapped_column(String(20), nullable=False)
    amount_orig: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)

    coin: Mapped['Coin'] = relationship(back_populates='transaction')
    user: Mapped['User'] = relationship(backref='transaction')

    def set_status(self, new_status: Literal["open", "cancel", "approve"]) -> None:

        assert new_status in ["open", "cancel", "approve"], "Invalid status"

        self.status = new_status


class News(Base):
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(50), default="news")
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    text: Mapped[str] = mapped_column(String(1000), nullable=False)
    date: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    news_coin: Mapped[list['Coin']] = relationship(
        'Coin', 
        secondary="news_coins",
        back_populates='news'
    )


class NewsHistoryCoin(Base):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id_news: Mapped[int] = mapped_column(ForeignKey('newss.id'))
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))
    score: Mapped[float] = mapped_column(Float, default=0)
    news_score_global: Mapped[Integer] = mapped_column(Integer, default=100)

    # news: Mapped['News'] = relationship(back_populates='history_coins')
    # coin: Mapped['Coin'] = relationship(back_populates='history_coins')