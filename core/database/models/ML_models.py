# модели для БД
from sqlalchemy import ForeignKey, Float, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base


class AgentAction(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey('agents.id'))
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    loss: Mapped[float] = mapped_column(Float, nullable=False)

    agent: Mapped['Agent'] = relationship(back_populates='actions')


class StatisticAgent(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey('agents.id'))

    type: Mapped[str] = mapped_column(String(50), nullable=False)
    loss: Mapped[float] = mapped_column(Float, nullable=False)

    agent: Mapped['Agent'] = relationship(back_populates='stata')


class Agent(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    path_model: Mapped[str] = mapped_column(String(100), unique=True)
    a_conficent: Mapped[float] = mapped_column(Float, default=0.95)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    version: Mapped[str] = mapped_column(String(20), default="0.0.1")

    actions: Mapped[list['AgentAction']] = relationship(
        back_populates='agent',
    )

    strategies: Mapped[list['Strategy']] = relationship(
        secondary="strategy_agents",
        back_populates='agents'
    )

    stata: Mapped[list['StatisticAgent']] = relationship(
        back_populates='agent'
    )


class ModelAction(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('m_l__models.id'))
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    loss: Mapped[float] = mapped_column(Float, default=1)

    model: Mapped['ML_Model'] = relationship(back_populates='actions')

class StatisticModel(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('m_l__models.id'))

    type: Mapped[str] = mapped_column(String(50), nullable=False)
    loss: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1score: Mapped[float] = mapped_column(Float, nullable=False)

    model: Mapped['ML_Model'] = relationship(back_populates='stata')

class ML_Model(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    path_model: Mapped[str] = mapped_column(String(100), unique=True)

    version: Mapped[str] = mapped_column(String(20), default="0.0.1")

    actions: Mapped[list['ModelAction']] = relationship(
        back_populates='model'
    )

    stata: Mapped[list['StatisticModel']] = relationship(
        back_populates='model'
    )
