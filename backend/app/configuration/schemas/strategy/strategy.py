from typing import Literal
from pydantic import BaseModel


class StrategyResponse(BaseModel):
    id: int
    type: Literal["train", "test", "trade"]

class CreateStrategyResponse(StrategyResponse):
    pass
