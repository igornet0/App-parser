# модели для БД
from sqlalchemy import ForeignKey, Float, String, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Literal
from core.database.base import Base


class FeatureArgument(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    feature_id: Mapped[int] = mapped_column(ForeignKey('features.id'))
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)

    feature: Mapped['Feature'] = relationship(back_populates='arguments')


class Feature(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)

    arguments: Mapped[list['FeatureArgument']] = relationship(
        back_populates='feature'
    )

    features: Mapped[list['AgentFeature']] = relationship(
        back_populates='agent_feature'
    )

    
class Agent(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(50), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    path_model: Mapped[str] = mapped_column(String(100), unique=True)
    a_conficent: Mapped[float] = mapped_column(Float, default=0.95)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    version: Mapped[str] = mapped_column(String(20), default="0.0.1")

    status: Mapped[str] = mapped_column(String(20), default="open")

    actions: Mapped[list['AgentAction']] = relationship(
        back_populates='agent',
    )

    features: Mapped[list['AgentFeature']] = relationship(
        back_populates='agent'
    )

    strategies: Mapped[list['Strategy']] = relationship(
        secondary="strategy_agents",
        back_populates='agents'
    )

    stata: Mapped[list['StatisticAgent']] = relationship(
        back_populates='agent'
    )

    trains: Mapped[list['AgentTrain']] = relationship(
        back_populates='agent'
    )

    def set_status(self, status: Literal["train", "stop", "open"]) -> None:

        assert status in ["train", "stop", "open"], "Invalid status, use train, stop or open"
        
        self.status = status

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


class AgentFeature(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey('agents.id'))
    feature_id: Mapped[int] = mapped_column(ForeignKey('features.id'))

    feature_value: Mapped[list['AgentFeatureValue']] = relationship(
        back_populates='agent_feature'
    )

    agent_feature: Mapped['Feature'] = relationship(back_populates='features')
    agent: Mapped['Agent'] = relationship(back_populates='features')


class AgentFeatureValue(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_feature_id: Mapped[int] = mapped_column(ForeignKey('agent_features.id'))
    feature_name: Mapped[str] = mapped_column(String(50), nullable=False)

    value: Mapped[float | str | int] = mapped_column(JSON)

    agent_feature: Mapped['AgentFeature'] = relationship(back_populates='feature_value')


class AgentType(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)


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

class ModelAction(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('m_l__models.id'))
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    loss: Mapped[float] = mapped_column(Float, default=1)

    model: Mapped['ML_Model'] = relationship(back_populates='actions')

class ModelType(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)

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