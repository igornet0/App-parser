import torch
import torch.nn as nn
import torch.nn.functional as F

class OrderDecisionModel(nn.Module):
    def __init__(
        self,
        price_window_size=50,
        hidden_size=128,
        num_lstm_layers=2
    ):
        super().__init__()
        self.price_window_size = price_window_size
        
        # Модуль обработки временного ряда цен
        self.price_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Модуль обработки фундаментальных факторов
        self.fundamental_fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        
        # Комбинированный классификатор
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 3)
        )

    def forward(self, inputs):
        # Распаковка входных данных
        price_window = inputs['price_window']  # [batch, window_size]
        current_price = inputs['current_price']  # [batch]
        risk = inputs['risk']  # [batch]
        position_size = inputs['position_size']  # [batch]
        action_probs = inputs['action_probs']  # [batch, 3]
        news_score = inputs['news_score']  # [batch]
        
        # Обработка ценового окна
        price_window = price_window.unsqueeze(-1)  # [batch, window_size, 1]
        lstm_out, _ = self.price_lstm(price_window)
        price_features = lstm_out[:, -1, :]  # Последний скрытый состояние
        
        # Обработка фундаментальных факторов
        fundamentals = torch.cat([
            current_price.unsqueeze(1),
            risk.unsqueeze(1),
            position_size.unsqueeze(1),
            action_probs,
            news_score.unsqueeze(1)
        ], dim=1)
        
        fundamental_features = self.fundamental_fc(fundamentals)
        
        # Комбинирование признаков
        combined = torch.cat([price_features, fundamental_features], dim=1)
        
        # Классификация решения
        logits = self.combined_fc(combined)
        return F.softmax(logits, dim=1)