import torch
import copy
import math
from typing import Dict, Any, Union, Generator, Tuple
import pandas as pd

from backend.Dataset.indicators import Indicators

import logging
logger = logging.getLogger("MMM.Agent")

class Agent:

    """
    Agent is a subclass of the Agent class that represents a specific type of agent
    designed to read time series data from a specified source.

    Attributes:
        name (str): The name of the agent.
        indecaters (dict): A dictionary containing indecater information.
        timetravel (str): The time travel parameter for the agent.
        discription (str): A description of the agent.
        model_parameters (dict): Additional parameters for the agent.
    """

    model = None

    def __init__(self, name: str, indecaters: Dict[str, Dict[str, Any]], timetravel: str = "5m",
                 discription: str = "Agent", model_parameters: Dict[str, Any] = {},
                 shema_RP: Dict[str, Generator[None, None, Any]] = {}):
        """
        mod_RP - Random Parameters in Indecaters
        """

        self.id = 1
        self.name = name
        self.indecaters = copy.deepcopy(indecaters)
        if "RP" in self.name:
            self.indecaters = self.replace_question_marks(shema_RP, self.indecaters)
        
        self.timetravel = timetravel
        self.discription = discription
        self._model_parameters = model_parameters
        self.lr_factor = 1

        self._init_model(self.model_parameters)

    def set_id(self, id: int):
        self.id = id

    def _init_model(self, model_parameters: Dict[str, Any]) -> None:
        return None

    def get_name(self) -> str:
        return self.name
    
    def get_timetravel(self) -> str:
        return self.timetravel
    
    def set_indicators(self, indicators: Dict[str, Dict[str, Any]]):
        self.indecaters = indicators

    @staticmethod
    def replace_question_marks(schema: Dict[str, Generator[None, None, Any]], 
                               indicators: Dict[str, Dict[str, Any]]):
        new_indicators = copy.deepcopy(indicators)

        for indicator in new_indicators.values():
            for param_name in indicator:
                if indicator[param_name] == '?':
                    try:
                        indicator[param_name] = next(schema[param_name])
                    except KeyError:
                        logger.error(f"Parameter {param_name} not found in schema")
    
        return new_indicators    
    
    def get_discription(self) -> str:
        return self.discription
    
    @property  
    def model_parameters(self) -> Dict[str, Any]:
        return self._model_parameters

    def get_indecaters(self) -> Dict[str, Dict[str, Any]]:
        return self.indecaters
    
    def get_shape_indecaters(self) -> Dict[str, int]:
        shapes = {}
        for indecater_name, _ in self.indecaters.items():
            shapes[indecater_name] = Indicators.get_shape(indecater_name)

        return shapes
    
    def get_datetime_format(self) -> Union[str, None]:
        return self.model_parameters.get("datetime_format", "")
    
    def get_count_input_features(self) -> int:
        return sum([sum(self.get_shape_indecaters().values()), 
                    self.get_datetime_format().count("%"),
                    len(self.model_parameters["input_features"])])

    def get_count_output_features(self) -> int:
        return len(self.model_parameters["output_features"])
    
    def get_model(self):
        return self.model
    
    @staticmethod
    def _prepare_datetime(data: pd.DataFrame) -> pd.DataFrame:

        data["year"] = pd.to_datetime(data["datetime"]).dt.year
        data["month"] = pd.to_datetime(data["datetime"]).dt.month
        data["day"] = pd.to_datetime(data["datetime"]).dt.day
        data["hour"] = pd.to_datetime(data["datetime"]).dt.hour
        data["minute"] = pd.to_datetime(data["datetime"]).dt.minute
        data["second"] = pd.to_datetime(data["datetime"]).dt.second
        data["weekday"] = pd.to_datetime(data["datetime"]).dt.strftime("%w")

        data["year"] = data["year"].astype(int)
        data["month"] = data["month"].astype(int)
        data["day"] = data["day"].astype(int)
        data["hour"] = data["hour"].astype(int)
        data["minute"] = data["minute"].astype(int)
        data["second"] = data["second"].astype(int)
        data["weekday"] = data["weekday"].astype(int)

        data["weekday"] = data["weekday"].apply(lambda x: 7 if x == 0 else x)

        data.drop("datetime", axis=1, inplace=True)

        return data

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        for indecater_name, params in self.get_indecaters().items():
            data = Indicators.calculate(indecater_name, data.copy(), **params)
        
        data = self._prepare_datetime(data)
        
        # column_time = "datetime"
        column_time = ["month", "day", "hour", "minute", "weekday"]

        drop_columns = ["year", "second"]

        time_features = data[column_time]

        data = data.drop(column_time, axis=1)
        data = data.drop(drop_columns, axis=1)

        return data, time_features
    
    def set_model(self, model):
        self.model = model

    def get_trade(self, data):
        if self.model is None:
            raise ValueError("Model is not set. Please set the model before calling get_trade.")
        
        return self.model.predict(data)
