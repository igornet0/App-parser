from pydantic import BaseModel

class CoinResponse(BaseModel):
    id: int
    name: str
    price_now: float