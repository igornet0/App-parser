from typing import Dict, Any, Tuple, List, Generator
import pandas as pd
from torch.amp import autocast, GradScaler
import numpy as np
import torch
import json

from backend.Dataset.indicators import Indicators

from .agent import Agent
from .models import PricePredictorModel

class AgentPReadTime(Agent):
    
    model = PricePredictorModel

    target_column = ["close"]

    def _init_model(self, model_parameters: Dict[str, Any]) -> PricePredictorModel:
        """
        Initializes the model for the agent.

        Args:
            model_parameters (dict): The configuration model for the agent containing parameters such as
                input features, sequence length, prediction length, model dimension, number of heads,
                and dropout rate.

        Returns:
            PricePredictorModel: An instance of the PricePredictorModel class.

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

        self.model = PricePredictorModel(pred_len=pred_len,
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
    
    def init_model_to_train(self, base_lr, weight_decay, 
                            is_cuda, effective_mp,
                            patience):
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

        return optimizer, scheduler, scaler
    
    def create_time_line_loader(self, data: pd.DataFrame, pred_len, seq_len) -> Generator[None, None, Tuple]:

        data, y, time_features = self.preprocess_data_for_model(data)

        n_samples = data.shape[0]

        for i in range(n_samples - pred_len - seq_len):
            new_x = data[i:i+seq_len]
            new_y = y[i+seq_len: i + seq_len + pred_len]
            time_x = time_features[i:i+seq_len]

            yield new_x, new_y, time_x
    
    def preprocess_data_for_model(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        data = super().preprocess_data_for_model(data)

        column_time = self.get_column_time()

        drop_columns = ["second"]
        
        if self.mod == "trade":
            data = data[-self.model_parameters["seq_len"]:]

        time_features = data[column_time]

        if self.mod == "test":
            tatget = self.procces_target(self.mod, data, self.target_column)
            return [data, tatget, time_features]

        column_time.extend(drop_columns)

        data = data.drop(column_time, axis=1)

        column_output = self.get_column_output()

        data = data[column_output]

        if self.mod == "train":
            tatget = self.procces_target(self.mod, data, self.target_column)
            return [data.values, tatget, time_features.values]
        
        bath = [data.values, time_features.values]

        return self.process_batch(bath)

    @staticmethod
    def procces_target(mod, data: pd.DataFrame, target_column) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Target must be a pandas DataFrame.")
        
        if mod == "test":
            target_column_new = ["datetime"]
            target_column_new.extend(target_column)
            return data[target_column_new]
        
        return data[target_column].values
    
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

    def save_json(self, epoch, history_loss, best_loss, base_lr, batch_size, weight_decay, filename):
        training_info = {
            'epochs_trained': epoch + 1,
            'loss_history': history_loss,
            'best_loss': best_loss,
            'indecaters': self.get_indecaters(),
            "seq_len": self.model_parameters["seq_len"],
            "pred_len": self.model_parameters["pred_len"],
            "d_model": self.model_parameters.get("d_model", 128),
            "n_heads": self.model_parameters.get("n_heads", 4),
            "emb_month_size": self.model_parameters.get("emb_month_size", 8),
            "emb_weekday_size": self.model_parameters.get("emb_weekday_size", 4),
            "lstm_hidden": self.model_parameters.get("lstm_hidden", 256),
            "num_layers": self.model_parameters.get("num_layers", 2),
            "dropout": self.model_parameters.get("dropout", 0.2),
            'hyperparams': {
                'base_lr': base_lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay
            }
        }
    
        with open(filename, 'w') as f:
            json.dump(training_info, f, indent=2)
    
    def loss_function(self, y_pred, y_true):
        y_true = y_true.squeeze(dim=-1) 
        return self.model.loss_function(y_pred, y_true)
