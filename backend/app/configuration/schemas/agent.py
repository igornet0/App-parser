from pydantic import BaseModel
from enum import Enum as PyEnum
import uuid
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr
from datetime import datetime

class AgentType(str, PyEnum):

    PREDTIME = 'agent_predtime'
    TRADETIME = 'agent_tradetime'
    NEWS = 'agent_news'

class DataTrade(str, PyEnum):
    pass


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

class AgentCreate(BaseModel):
    name: str
    type: AgentType
    path_model: str
    a_conficent: float = 0.95
    active: bool = True
    version: str = "0.0.1"

class AgentResponse(BaseModel):
    id: int
    name: str = None
    type: AgentType = None

    path_model: str = None
    a_conficent: float = 0.95
    active: bool = True
    version: str = "0.0.1"


