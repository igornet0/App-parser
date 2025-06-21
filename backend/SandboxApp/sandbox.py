import asyncio

from typing import List, Dict, Generator, Any, Union, Literal

from backend.Dataset import DatasetTimeseries
from backend.train_models.transform_data import TimeSeriesTransform
from core.database import orm_get_coins, orm_get_timeseries_by_coin, orm_get_data_timeseries
from core.database import db_helper

from .boxs import Box, Exhange

import logging 

logger = logging.getLogger("Sandbox")

class Sandbox:

    type_box = {
        "Box": Box,
        "Exhange": Exhange
    }

    def __init__(self, data: List[DatasetTimeseries] = [], 
                 agents: List[Any] = [], 
                 db_use: bool = False):
        
        if db_use:
            self.data = asyncio.run(self._load_data_fron_db())
            self.agents = asyncio.run(self._load_agents_from_db())
        else:
            self.data = data
            self.agents = agents

        self._box = None

    @classmethod
    def create_box(cls, type_box: Literal["Box", "Exhange"], **kwargs) -> Union[Box, Exhange]:
        if cls.type_box.get(type_box) is None:
            raise ValueError(f"Unknown box type: {type_box}")
        
        box = cls.type_box[type_box](**kwargs)

        return box
    
    async def _load_data_fron_db(self):
        results = {}
        async with db_helper.get_session() as session:
            coins = await orm_get_coins(session)
            for coin in coins:
                logger.info(f"Loading data for coin: {coin.name}")
                timeseries = await orm_get_timeseries_by_coin(session, coin)
                for ts in timeseries:
                    logger.info(f"Loading data for timeseries: {ts.id}")
                    result = await orm_get_data_timeseries(session, ts.id)
                    dt = DatasetTimeseries(result)
                    results.setdefault(coin.name, {})
                    results[coin.name][ts.timestamp] = dt

        return results

    async def _load_agents_from_db(self):
        pass

    def start(self):
        pass

    def add_data(self, item):
        self.data.append(item)

    def get_data(self):
        return self.data

    def clear_data(self):
        self.data = []