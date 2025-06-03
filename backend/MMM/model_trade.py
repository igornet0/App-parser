import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingModel(nn.Module):
    def __init__(self, 
                 input_size=13, 
                 pred_size=6, 
                 time_size=5, 
                 hidden_size=128,
                 num_layers=2,
                 output_size=3,
                 dropout=0.3):
        super(TradingModel, self).__init__()
        total_features = input_size + pred_size + time_size
        
        # Основной LSTM для обработки временных рядов
        self.lstm = nn.LSTM(
            input_size=total_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Дополнительные полносвязные слои
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, output_size)
        
        # Нормализация
        self.layer_norm = nn.LayerNorm(total_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_pred, time_data, return_probs=True):
        # Объединение всех входных данных
        combined = torch.cat([
            x, 
            x_pred, 
            time_data
        ], dim=-1)
        
        # Нормализация и регуляризация
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        # Обработка LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Берем только последний временной шаг
        last_output = lstm_out[:, -1, :]
        
        # Дополнительная обработка
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Выходной слой
        logits = self.output(x)
        
        if return_probs:
            # Преобразование в вероятности действий (0-100)
            probs = F.softmax(logits, dim=-1) * 100
            
            return probs
        
        return logits