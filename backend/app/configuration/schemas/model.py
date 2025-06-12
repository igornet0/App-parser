from pydantic import BaseModel

class ModelAction(BaseModel):
    id: int
    action: str
    loss: float

class Agent(BaseModel):
    id: int
    type: str
    path_model: str
    
    version: str

    actions: list[ModelAction]


