from typing import Dict, Any, Tuple
import pandas as pd
import torch

from backend.Dataset.indicators import Indicators

from .agent import Agent
from ..models import RiskAwareSACNetwork

class AgentRisk(Agent):

    _type = "AgentRisk"
    
    model = RiskAwareSACNetwork

    target_column = ["close"]

    def _init_model(self, model_parameters: Dict[str, Any]) -> RiskAwareSACNetwork:
        """
        Initializes the model for the agent.

        Args:
            model_parameters (dict): The configuration model for the agent containing parameters such as
                input features, sequence length, prediction length, model dimension, number of heads,
                and dropout rate.

        Returns:
            RiskAwareSACNetwork: An instance of the RiskAwareSACNetwork class.

        """
        # n_indicators = sum(self.get_shape_indecaters().values())
        # input_features = model_parameters.get("input_features", ['close', 'max', 'min', 'volume'])
        input_features = self.get_count_input_features()
        seq_len = model_parameters["seq_len"]
        pred_len = model_parameters["pred_len"]
        d_model = model_parameters.get("d_model", 128)
        n_heads = model_parameters.get("n_heads", 4)

        emb_month_size = model_parameters.get("emb_month_size", 8)
        emb_weekday_size = model_parameters.get("emb_weekday_size", 4)

        lstm_hidden = model_parameters.get("lstm_hidden", 256)
        num_layers = model_parameters.get("num_layers", 2)
        dropout = model_parameters.get("dropout", 0.2)

        self.model = RiskAwareSACNetwork(pred_len=pred_len,
                                    seq_len=seq_len,
                                    num_features=input_features,
                                    n_heads=n_heads,
                                    d_model=d_model,
                                    emb_month_size=emb_month_size, 
                                    emb_weekday_size=emb_weekday_size, 
                                    lstm_hidden=lstm_hidden, 
                                    num_layers=num_layers, 
                                    dropout=dropout)

        return self.model
    
    def preprocess_data_for_model(self, data: pd.DataFrame, normalize=False) -> Tuple[pd.DataFrame, pd.DataFrame]:

        for indecater_name, params in self.get_indecaters().items():
            data = Indicators.calculate(indecater_name, data.copy(), **params)

        if normalize:
            data = self.normalize_data(data)

        data = data.dropna()
        
        data = self._prepare_datetime(data)
        
        column_time = ["month", "day", "hour", "minute", "weekday"]

        drop_columns = ["second"]

        time_features = data[column_time]

        column_time.extend(drop_columns)

        data = data.drop(column_time, axis=1)

        column_output = []
        column_output.extend(self.model_parameters["input_features"])

        for indecater_name, params in self.get_indecaters().items():
            indecater = Indicators.collumns_shape.get(indecater_name)
            collumn = Indicators.paser_collumn_name(indecater, **params)
            
            if isinstance(collumn, list):
                column_output.extend(collumn)
            else:
                column_output.append(collumn)

        data = data[column_output]

        return data, time_features
    
    def procces_target(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Target must be a pandas DataFrame.")
        
        return data[self.target_column]
    
    def save_model(self, epoch, optimizer, scheduler, best_loss, filename: str):
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        torch.save({
            'epoch': epoch,
            'indecaters': self.get_indecaters(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            "seq_len": self.model_parameters["seq_len"],
            "pred_len": self.model_parameters["pred_len"],
            "d_model": self.model_parameters.get("d_model", 128),
            "n_heads": self.model_parameters.get("n_heads", 4),
            "emb_month_size": self.model_parameters.get("emb_month_size", 8),
            "emb_weekday_size": self.model_parameters.get("emb_weekday_size", 4),
            "lstm_hidden": self.model_parameters.get("lstm_hidden", 256),
            "num_layers": self.model_parameters.get("num_layers", 2),
            "dropout": self.model_parameters.get("dropout", 0.2),
            'loss': best_loss,
        }, filename)
    
    def loss_function(self, y_pred, y_true):
        y_true = y_true.squeeze(dim=-1) 
        return self.model.loss_function(y_pred, y_true)
