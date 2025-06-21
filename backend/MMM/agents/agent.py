import torch
import copy
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union, Generator, Tuple, Literal, List, Union
import numpy as np
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
    _type = "Agent"
    model = None
    mod: Literal["train", "trade", "test"] = "test"

    input_features = []

    model_parameters_default = {}

    def __init__(self, name: str = "Agent", indecaters: Dict[str, Dict[str, Any]] = {}, timetravel: str = "5m",
                 discription: str = "Agent", model_parameters: Dict[str, Any] = {},
                 data_normalize: bool = False,
                 shema_RP: Dict[str, Generator[None, None, Any]] = {},
                 RP_I: bool = False):
        """
        mod_RP - Random Parameters in Indecaters
        """

        self.id = 1
        self.name = name
        self.version = 1
        self.indecaters = copy.deepcopy(indecaters)

        if "RP" in self.name:
            self.indecaters = self.replace_question_marks(shema_RP, self.indecaters)

        if RP_I:
            self.indecaters = self.indecater_with_random()
        
        self.timetravel = timetravel
        self.discription = discription
        self.data_normalize = data_normalize

        if len(model_parameters) == 0:
            model_parameters = self.model_parameters_default
        elif len(model_parameters) < len(self.model_parameters_default):
            model_parameters.update(copy.copy(self.model_parameters_default))

        self._model_parameters = model_parameters

        self.optimizer_state_dict = None
        self.scheduler_state_dict = None

        self.lr_factor = 1
        self.history = []
        self.data_buffer = []
        self.data_buffer_size = 1000

        self._init_model(self._model_parameters)

    @classmethod
    def get_type(cls):
        return cls._type

    def set_id(self, id: int):
        self.id = id

    def set_mode(self, new_mode: Literal["train", "trade", "test"]):
        self.mod = new_mode

    def indecater_with_random(self):
        new_indicators = {}
        for name, indicator in self.indecaters.items():
            if torch.randint(0, 2, (1, 1)).item() == 1:
                new_indicators[name] = indicator

        return new_indicators
    
    def create_time_line_loader(self, data: pd.DataFrame, pred_len, seq_len) -> Generator[None, None, Tuple]:

        data = self.preprocess_data_for_model(data)

        n_samples = data.shape[0]

        for i in range(n_samples - pred_len - seq_len):
            new_x = data[i:i+seq_len]

            yield new_x,

    def _init_model(self, model_parameters: Dict[str, Any]) -> None:
        return None

    def get_name(self) -> str:
        return self.name
    
    def get_timetravel(self) -> str:
        return self.timetravel
    
    def set_indicators(self, indicators: Dict[str, Dict[str, Any]]):
        self.indecaters = indicators

    def get_indecaters_column(self) -> List[str]:
        column_output = []
        for indecater_name, params in self.get_indecaters().items():
            indecater = Indicators.collumns_shape.get(indecater_name)
            collumn = Indicators.paser_collumn_name(indecater, **params)
            
            if isinstance(collumn, list):
                column_output.extend(collumn)
            else:
                column_output.append(collumn)
        
        return column_output
    
    def get_indecaters_dict(self) -> Dict[str, Union[List, str]]:
        column_output = {}
        for indecater_name, params in self.get_indecaters().items():
            indecater = Indicators.collumns_shape.get(indecater_name)
            collumn = Indicators.paser_collumn_name(indecater, **params)
            
            column_output[indecater_name] = collumn
        
        return column_output

    def get_column_output(self) -> List[str]:
        column_output = []
        column_output.extend(self.input_features)
        column_output.extend(self.get_indecaters_column())
        
        return column_output

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
        return self.model_parameters.get("datetime_format", "%m-%d %H:%M %w")
    
    def get_column_time(self) -> List[str]:
        dt_format = self.get_datetime_format()
        dt_format = dt_format.split("%")

        def clear_fromat(x):
            for c in [" ", "-", ":"]:
                x = x.replace(c, "")
            return x

        dt_format = map(clear_fromat, dt_format)

        dt_key = {"Y": "year", 
                  "m": "month", 
                  "d": "day", 
                  "H": "hour", 
                  "M":"minute", 
                  "w": "weekday"}
        
        return [dt_key[key] for key in dt_format if key in dt_key]
    
    def get_count_input_features(self) -> int:
        return sum([sum(self.get_shape_indecaters().values()), 
                    self.get_datetime_format().count("%"),
                    len(self.input_features)])

    def get_count_output_features(self) -> int:
        return len(self.model_parameters["output_features"])
    
    def init_model_to_train(self, base_lr, weight_decay, 
                            is_cuda, effective_mp,
                            patience):
        pass

    def set_version(self, version: int):
        self.version = version

    def load_model(self, optimizer, scheduler, filename: str):
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        checkpoint = torch.load(filename)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load additional information
        epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']

        return epoch, best_loss
    
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

        # cols = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday']
        # data.loc[:, cols] = data[cols].astype(int)

        data["year"] = data["year"].astype(int)
        data["month"] = data["month"].astype(int)
        data["day"] = data["day"].astype(int)
        data["hour"] = data["hour"].astype(int)
        data["minute"] = data["minute"].astype(int)
        data["second"] = data["second"].astype(int)
        data["weekday"] = data["weekday"].astype(int)

        data["weekday"] = data["weekday"].apply(lambda x: 7 if x == 0 else x)

        return data
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:

        for indecater_name, params in self.get_indecaters().items():
            data = Indicators.calculate_normalized(indecater_name, data.copy(), **params)

        data['close'] = (data['close'] - data['open']) / data['open'] * 100
        data['max'] = (data['max'] - data['open']) / data['open'] * 100
        data['min'] = (data['min'] - data['open']) / data['open'] * 100

        return data

    def preprocess_data_for_model(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        collumn = data.columns
        dd = data.copy()

        new_data = data.copy()
        for indecater_name, params in self.get_indecaters().items():
            new_data = Indicators.calculate(indecater_name, new_data, **params)

        if self.data_normalize:
            new_data = self.normalize_data(new_data)

        new_data = new_data.dropna()

        if len(new_data) == 0:
            print(f"Data columns: {data.columns} - {collumn}")
            print(dd)
            print(data)
            raise ValueError("Data is empty")
        # else:
        #     print(f"Data shape: {new_data.shape}")
        #     print(dd)
        #     print(f"Data columns: {new_data.columns}")
        
        data = self._prepare_datetime(new_data)

        if self.mod == "test":
            return data
        
        data.drop("datetime", axis=1, inplace=True)

        return data
    
    def procces_target(self, target: pd.DataFrame) -> pd.DataFrame:
        if isinstance(target, pd.Series):
            target = target.to_frame()

        if "close" in target.columns:
            target["close"] = (target["open"] - target["close"]) / target["open"] * 100
            target["max"] = (target["open"] - target["max"]) / target["open"] * 100
            target["min"] = (target["open"] - target["min"]) / target["open"] * 100

        return target
    
    def process_batch(self, batch: List[np.ndarray]):
        # Векторизованная обработка батча
        tranfor_data = lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32)
        # wraper_nath = lambda data: map(tranfor_data, data)
        # return list(map(wraper_nath, zip(*batch)))
        
        return list(map(tranfor_data, zip(*batch)))
            
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
    
    def trade(self, data: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model is not set. Please set the model before calling trade.")
        
        self.update_data_buffer(data)

        if self.mod == "trade":
            data = self.preprocess_data_for_model(data)

        action = self.model(*data)
        self.update_history(action, data)

        return action
    
    def get_str_indecaters(self):
        result = ""
        for indecater_name, params in self.get_indecaters().items():
            result += f"\t{indecater_name}: {params}\n"

        return result
    
    def get_filename_pth(self) -> Path:
        from core import data_helper
        version_s = ".".join([*str(self.version)])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        filename = data_helper["models pth"] / self.get_type() / f"{self.name}_{self.timetravel}_{version_s}_{timestamp}.pth"
        return filename
    
    def get_filename_json(self) -> Path:
        from core import data_helper
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        filename = data_helper["models logs"] / self.get_type() / f"{self.name}_{self.timetravel}_training_log_{timestamp}.json"
        return filename
    
    def save_model(self, epoch, optimizer, scheduler, best_loss):
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        filename = self.get_filename_pth()

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
        }, filename)
    
    def save_json(self, epoch, history_loss, best_loss, base_lr, batch_size, weight_decay):
        training_info = {
            'epochs_trained': epoch + 1,
            'loss_history': history_loss,
            'best_loss': best_loss,
            'indecaters': self.get_indecaters(),
            'hyperparams': {
                'base_lr': base_lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay
            }
        }

        filename = self.get_filename_json()
    
        with open(filename, 'w') as f:
            json.dump(training_info, f, indent=2)
    
    def __str__(self):
        return f"Agent: {self.name} - {self.discription}\nIndecaters:\n{self.get_str_indecaters()}"
