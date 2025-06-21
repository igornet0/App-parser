from typing import Dict, Any, Tuple, List, Generator, Union
import pandas as pd
import torch
from torch.amp import GradScaler
import torch.nn as nn
import json

from .agent import Agent
from ..models import TradingModel

class AgentTradeTime(Agent):

    _type = "AgentTradeTime"
    
    model = TradingModel

    criterion = nn.HuberLoss

    input_features = ["close", "max", "min", "volume"]

    model_parameters_default = {
        "seq_len": 50,
        "pred_len": 5,
        "hidden_size": 128,
        "n_heads": 8,
        "emb_month_size": 8,
        "emb_weekday_size": 4,
        "lstm_hidden": 256,
        "num_layers": 4,
        "dropout": 0.4,
    }

    def __init__(self, name: str = "Agent", indecaters: Dict[str, Dict[str, Any]] = {}, timetravel: str = "5m",
                 discription: str = "Agent", model_parameters: Dict[str, Any] = {},
                 data_normalize: bool = False,
                 shema_RP: Dict[str, Generator[None, None, Any]] = {},
                 RP_I: bool = False, 
                 proffit_preddict_for_buy=0.9, 
                 proffit_preddict_for_sell=0.9):
        
        self.proffit_preddict_for_buy = proffit_preddict_for_buy
        self.proffit_preddict_for_sell = proffit_preddict_for_sell

        super().__init__(name, indecaters, timetravel, discription, model_parameters, data_normalize, shema_RP, RP_I)

    def _init_model(self, model_parameters: Dict[str, Any]) -> TradingModel:
        """
        Initializes the model for the agent.

        Args:
            model_parameters (dict): The configuration model for the agent containing parameters such as
                input features, sequence length, prediction length, model dimension, number of heads,
                and dropout rate.

        Returns:
            TradingModel: An instance of the TradingModel class.

        """

        # input_size = self.get_count_input_features() - self.get_datetime_format().count("%")

        self.proffit_preddict_for_buy = self.proffit_preddict_for_buy
        self.proffit_preddict_for_sell = self.proffit_preddict_for_sell
        # print("input_size=", self.get_count_input_features())
        self.model = TradingModel(
            input_size=self.get_count_input_features(),
            **model_parameters
        )

        return self.model
    
    def init_model_to_train(self, base_lr, weight_decay, 
                            is_cuda, effective_mp,
                            patience):

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion(delta=0.7)

        # Оптимизатор и планировщик
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr * self.lr_factor,
            weight_decay=weight_decay,
            fused=is_cuda
        )

        # Инициализация GradScaler только при необходимости
        scaler = GradScaler(enabled=effective_mp)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            "min",
            factor=0.5,
            patience=patience
        )

        if self.scheduler_state_dict is not None:
            scheduler.load_state_dict(self.scheduler_state_dict)
            

        return optimizer, scheduler, scaler

    def create_time_line_loader(self, data: pd.DataFrame, pred_len, seq_len) -> Generator[None, None, Tuple]:

        data, target_time, tatget, time_features = self.preprocess_data_for_model(data)

        n_samples = data.shape[0]

        for i in range(n_samples - pred_len - seq_len):
            new_x = data[i:i+seq_len]
            new_y = tatget[i+seq_len: i + seq_len + pred_len]
            new_target_time = target_time[i+seq_len: i + seq_len + pred_len]
            time_time = time_features[i:i+seq_len]

            yield new_x, new_y, new_target_time, time_time

    def preprocess_data_for_model(self, data: pd.DataFrame) -> Union[Tuple[pd.DataFrame, pd.DataFrame], List[torch.Tensor]]:

        data = super().preprocess_data_for_model(data)

        column_time = self.get_column_time()

        drop_columns = ["second"]
        
        if self.mod == "trade":
            data = data[-self.model_parameters["seq_len"]:]

        time_features = data[column_time]

        if self.mod == "test":
            target = self.procces_target(data)
            bath = [data, target, time_features]
            return bath

        column_time.extend(drop_columns)

        from .agent_pread_time import AgentPredTime
        target_pred_time = AgentPredTime.procces_target(self.mod, data.copy(), ["close"])

        new_data = data.drop(column_time, axis=1)

        column_output = self.get_column_output()

        new_data = new_data[column_output]

        if self.mod == "train":
            target = self.procces_target(data)
            bath = [new_data.values, target_pred_time, target, time_features.values]
            return bath
        
        bath = [new_data.values, target_pred_time, time_features.values]

        return self.process_batch(bath)
    
    def procces_target(self, data: pd.DataFrame) -> List[int]:

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Target must be a pandas DataFrame.")
        
        prices_close = data["close"].values

        targets = []

        for i in range(len(prices_close)):
            if prices_close[i] > self.proffit_preddict_for_buy:
                targets.append([100, 0, 0])
            elif abs(prices_close[i]) > self.proffit_preddict_for_sell:
                targets.append([0, 100, 0])
            else:
                targets.append([0, 0, 100])
        
        if self.mod == "test":
            data["target"] = targets

            return data

        return targets
    
    def trade(self, data: Union[List[pd.DataFrame], List[torch.Tensor]]):
        if self.model is None:
            raise ValueError("Model is not set. Please set the model before calling trade.")
        
        self.update_data_buffer(data)

        if self.mod == "trade":
            data = self.preprocess_data_for_model(data)

        elif self.mod == "train":
            data[1] = data[1].squeeze(dim=-1)

        action = self.model(*data)
        self.update_history(action, data)

        return action
    
    def save_model(self, epoch, optimizer, scheduler, best_loss):
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        filename = self.get_filename_pth()

        torch.save({
            'epoch': epoch,
            "name": self.name,
            "data_normalize": self.data_normalize,
            "timetravel": self.timetravel,
            'indecaters': self.get_indecaters(),
            "input_features": self.input_features,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            "datetime_format": self.get_datetime_format(),
            "seq_len": self.model_parameters["seq_len"],
            "pred_len": self.model_parameters["pred_len"],
            "hidden_size": self.model_parameters.get("hidden_size", 128),
            "n_heads": self.model_parameters.get("n_heads", 8),
            "num_layers": self.model_parameters.get("num_layers", 2),
            "emb_month_size": self.model_parameters.get("emb_month_size", 8),
            "emb_weekday_size": self.model_parameters.get("emb_weekday_size", 4),
            "lstm_hidden": self.model_parameters.get("lstm_hidden", 256),
            "dropout": self.model_parameters.get("dropout", 0.3),
            'loss': best_loss,
        }, filename)

    def save_json(self, epoch, history_loss, best_loss, base_lr, batch_size, weight_decay):
        training_info = {
            'epochs_trained': epoch + 1,
            'loss_history': history_loss,
            'best_loss': best_loss,
            "name": self.name,
            "data_normalize": self.data_normalize,
            "timetravel": self.timetravel,
            'indecaters': self.get_indecaters(),
            "input_features": self.input_features,
            "datetime_format": self.get_datetime_format(),
            "seq_len": self.model_parameters["seq_len"],
            "pred_len": self.model_parameters["pred_len"],
            "hidden_size": self.model_parameters.get("hidden_size", 128),
            "n_heads": self.model_parameters.get("n_heads", 8),
            "num_layers": self.model_parameters.get("num_layers", 2),
            "emb_month_size": self.model_parameters.get("emb_month_size", 8),
            "emb_weekday_size": self.model_parameters.get("emb_weekday_size", 4),
            "lstm_hidden": self.model_parameters.get("lstm_hidden", 256),
            "dropout": self.model_parameters.get("dropout", 0.3),
            'loss': best_loss,
            'hyperparams': {
                'base_lr': base_lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay
            }
        }

        filename = self.get_filename_json()
    
        with open(filename, 'w') as f:
            json.dump(training_info, f, indent=2)
    
    def loss_function(self, y_pred, y_true):
        # y_true = y_true.squeeze(dim=-1)
        return self.model.loss_function(self.criterion, y_pred, y_true)