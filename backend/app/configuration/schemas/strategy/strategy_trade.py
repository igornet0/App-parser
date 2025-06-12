from pydantic import BaseModel

class AgentAction(BaseModel):
    id: int
    action: str
    loss: float

class Agent(BaseModel):
    id: int
    name: str
    type: str

    path_model: str
    a_conficent: float
    active: bool
    version: str

    actions: list[AgentAction]


