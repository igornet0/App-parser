# модели для БД
from typing import Literal, List
from sqlalchemy import ForeignKey, Float, String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base

class StrategyCoin(Base):

    strategy_id: Mapped[int] = mapped_column(ForeignKey('strategys.id'), primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'), primary_key=True)


class StrategyAgent(Base):

    strategy_id: Mapped[int] = mapped_column(ForeignKey('strategys.id'), primary_key=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey('agents.id'), primary_key=True)


class Strategy(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    type: Mapped[str] = mapped_column(String(50), nullable=False) #test, trade real

    coins: Mapped[list['Coin']] = relationship(
        'Coin', 
        secondary="strategy_coins",
        back_populates='strategies'
    )

    agents: Mapped[list['Agent']] = relationship(
        secondary='strategy_agents',
        back_populates='strategies'
    )

    # Внешние ключи и отношения с условиями
    model_risk_id: Mapped[int] = mapped_column(
        ForeignKey('m_l__models.id'),
        nullable=True,
        comment="ID модели с типом 'RiskModel'"
    )
    model_order_id: Mapped[int] = mapped_column(
        ForeignKey('m_l__models.id'),
        nullable=True,
        comment="ID модели с типом 'OrderModel'"
    )

    model_risk: Mapped['ML_Model'] = relationship(
        'ML_Model',
        primaryjoin="and_(Strategy.model_risk_id == ML_Model.id, ML_Model.type == 'RiskModel')",
        foreign_keys=[model_risk_id],
        uselist=False
    )
    
    model_order: Mapped['ML_Model'] = relationship(
        'ML_Model',
        primaryjoin="and_(Strategy.model_order_id == ML_Model.id, ML_Model.type == 'OrderModel')",
        foreign_keys=[model_order_id],
        uselist=False
    )

    risk: Mapped[float] = mapped_column(Float, default=0.05)
    reward: Mapped[float] = mapped_column(Float, default=0.05)


class AgentTrain(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey('agents.id'))

    epochs: Mapped[int] = mapped_column(Integer, default=100)
    epoch_now: Mapped[int] = mapped_column(Integer, default=0)
    loss_now: Mapped[float] = mapped_column(Float, default=0)
    batch_size: Mapped[int] = mapped_column(Integer, default=64)
    learning_rate: Mapped[float] = mapped_column(Float, default=0.001)
    weight_decay: Mapped[float] = mapped_column(Float, default=0.001)

    status: Mapped[str] = mapped_column(String(20), default="start")

    coins: Mapped[List["Coin"]] = relationship(
        "Coin",  # Используем строковое имя класса
        secondary="train_coins",  # Имя таблицы ассоциаций
        back_populates="trains",  # Ссылка на обратное отношение в Coin
        viewonly=False
    )

    agent: Mapped["Agent"] = relationship(
        "Agent",
        back_populates="trains"
    )

    def set_status(self, new_status: Literal["start", "train", "stop"]):

        assert new_status in ["start", "train", "stop"], "Invalid status, use start, train or stop"

        self.status = new_status

class TrainCoin(Base):

    train_id: Mapped[int] = mapped_column(ForeignKey('agent_trains.id'), primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'), primary_key=True)