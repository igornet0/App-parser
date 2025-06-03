import torch
import copy
from typing import Dict, Any, Union, Generator, Tuple
import pandas as pd

from backend.Dataset.indicators import Indicators

import logging
logger = logging.getLogger("MMM.Agent")

HISTORY_SIZE = 1000

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
                 shema_RP: Dict[str, Generator[None, None, Any]] = {},
                 RM_I: bool = False):
        """
        mod_RP - Random Parameters in Indecaters
        """

        self.id = 1
        self.name = name
        self.indecaters = copy.deepcopy(indecaters)
        if "RP" in self.name:
            self.indecaters = self.replace_question_marks(shema_RP, self.indecaters)

        if RM_I:
            self.indecaters = self.indecater_with_random()
        
        self.timetravel = timetravel
        self.discription = discription
        self._model_parameters = model_parameters
        self.lr_factor = 1
        self.history = []
        self.data_buffer = []
        self.data_buffer_size = 1000

        self._init_model(self.model_parameters)

    def set_id(self, id: int):
        self.id = id

    def indecater_with_random(self):
        new_indicators = {}
        for name, indicator in self.indecaters.items():
            if torch.randint(0, 2, (1, 1)).item() == 1:
                new_indicators[name] = indicator

        return new_indicators

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
        return self._model_parameters.copy()

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
    
    def save_model(self, epoch, optimizer, scheduler, best_loss, filename: str):
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
        }, filename)
    
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
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:

        for indecater_name, params in self.get_indecaters().items():
            data = Indicators.calculate_normalized(indecater_name, data.copy(), **params)

        data['close'] = (data['close'] - data['open']) / data['open'] * 100
        data['max'] = (data['max'] - data['open']) / data['open'] * 100
        data['min'] = (data['min'] - data['open']) / data['open'] * 100

        return data

    def preprocess_data_for_model(self, data: pd.DataFrame, normalize=False) -> Tuple[pd.DataFrame, pd.DataFrame]:

        return data
    
    def procces_target(self, target: pd.DataFrame) -> pd.DataFrame:
        if isinstance(target, pd.Series):
            target = target.to_frame()

        if "close" in target.columns:
            target["close"] = (target["open"] - target["close"]) / target["open"] * 100
            target["max"] = (target["open"] - target["max"]) / target["open"] * 100
            target["min"] = (target["open"] - target["min"]) / target["open"] * 100

        return target
    
    def loss_function(self, y_pred, y_true):
        return self.model.loss_function(y_pred, y_true)
    
    def set_model(self, model):
        self.model = model

    def update_data_buffer(self, data):
        self.data_buffer.append(data)
        if len(self.data_buffer) > self.data_buffer_size:
            self.data_buffer.pop(0)

        return self.data_buffer
    
    def get_data_buffer(self):
        return self.data_buffer
    
    def update_history(self, action, data):
        self.history.append({"action": action, "data": data})

        if len(self.history) > HISTORY_SIZE:
            self.history.pop(0)

        return self.history

    def trade(self, data):
        if self.model is None:
            raise ValueError("Model is not set. Please set the model before calling trade.")
        
        self.update_data_buffer(data)

        action = self.model(*data)
        self.update_history(action, data)

        return action
    
    def get_str_indecaters(self):
        result = ""
        for indecater_name, params in self.get_indecaters().items():
            result += f"\t{indecater_name}: {params}\n"

        return result
    
    def __str__(self):
        return f"Agent: {self.name} - {self.discription}\nIndecaters:\n{self.get_str_indecaters()}"
