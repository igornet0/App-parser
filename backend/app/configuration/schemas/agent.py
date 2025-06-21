from pydantic import BaseModel
from enum import Enum as PyEnum
import uuid
from typing import Optional, Dict, Any
from backend.Dataset import Indicators
from pydantic import BaseModel, ConfigDict, EmailStr
from datetime import datetime

class AgentType(str, PyEnum):

    PREDTIME = 'AgentPredTime'
    TRADETIME = 'AgentTradeTime'
    NEWS = 'AgentNews'

class FeatureType(str, PyEnum):
    pass

class DataTrade(str, PyEnum):
    pass

class Argument(BaseModel):
    id: Optional[int] = 0
    name: Optional[str] = "Argument"
    type: Optional[str] = "int"
    volue: str | int | None = None

class FeatureTypeResponse(BaseModel):
    id: int
    name: str

    arguments: list[Argument]

class AgentTypeResponse(BaseModel):
    id: int
    name: str

class AgentTrade(BaseModel):
    id: int
    agent_id: int
    data: list

class AgentAction(BaseModel):
    id: int
    action: str
    loss: float

class AgentStrategy(BaseModel):
    id: int
    name: str

class AgentStata(BaseModel):
    id: int
    type: str
    loss: float

class FeatureParameters(BaseModel):
    # Динамический словарь для параметров индикатора
    # Пример: {"period": 14, "column": "close"}
    parameters: Dict[str, Any]

class Feature(BaseModel):
    id: int
    # feature_id: int
    name: Optional[str] = ""

    parameters: Dict[str, Any]

class AgentCreate(BaseModel):
    id: Optional[int] = None
    name: str
    type: AgentType
    status: Optional[str] = "open"

    path_model: Optional[str] = ""

    a_conficent: Optional[float] = 0.95
    active: Optional[bool] = True
    version: Optional[str] = "0.0.1"
    created: Optional[datetime] = None

    features: list[Feature] = []

class TrainData(BaseModel):
    class Config:
        orm_mode = True

    id: Optional[int] = None
    agent_id: Optional[int] = None
    status: Optional[str] = "start" 

    epochs: int
    epoch_now: Optional[int] = 0

    batch_size: int
    learning_rate: float
    weight_decay: float

class AgentTrainResponse(BaseModel):
    class Config:
        orm_mode = True

    id: Optional[int] = None
    name: str
    type: AgentType
    status: Optional[str] = "open"
    timeframe: Optional[str] = "5m"

    path_model: Optional[str] = ""

    a_conficent: Optional[float] = 0.95
    active: Optional[bool] = True
    version: Optional[str] = "0.0.1"
    created: Optional[datetime] = None

    features: list[Feature] = []
    coins: list[int] = []

    train_data: Optional[TrainData] = None

class AgentResponse(BaseModel):
    class Config:
        orm_mode = True

    id: Optional[int] = None
    name: str
    type: AgentType
    status: Optional[str] = "open"
    timeframe: Optional[str] = "5m"

    path_model: Optional[str] = ""

    a_conficent: Optional[float] = 0.95
    active: Optional[bool] = True
    version: Optional[str] = "0.0.1"
    created: Optional[datetime] = None

    features: list[Feature] = []


