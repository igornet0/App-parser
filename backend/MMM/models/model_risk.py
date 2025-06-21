import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketFeatureExtractor(nn.Module):
    """Извлекает временные паттерны из исторических данных"""
    def __init__(self, input_channels=5, features=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, features, batch_first=True)
        self.attention = nn.MultiheadAttention(features, num_heads=4, batch_first=True)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out[:, -1, :]


class RiskVolumePredictor(nn.Module):
    """Предсказывает риск и объем ордера"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.volume_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        risk = torch.sigmoid(self.risk_head(x))  # Диапазон [0, 1]
        volume_pct = torch.sigmoid(self.volume_head(x)) * 100  # Диапазон [0, 100%]
        return volume_pct, risk


class RiskAwareSACNetwork(nn.Module):
    """Модель для предсказания риска и объема"""
    
    criterion = nn.HuberLoss(delta=0.5)

    def __init__(
        self,
        historical_features=5,
        time_window=60,
        static_features=4,
        hidden_dim=128
    ):
        super().__init__()

        self.time_window = time_window

        # Модули для обработки рыночных и статических данных
        self.market_encoder = MarketFeatureExtractor(
            input_channels=historical_features,
            features=hidden_dim
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Объединение признаков и прогнозирование
        self.state_processor = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(2 * hidden_dim)
        )
        self.predictor = RiskVolumePredictor(2 * hidden_dim)
        
    def forward(self, market_data, static_data):
        # Проверка временного окна
        if market_data.size(1) != self.time_window:
            raise ValueError(f"Ожидается {self.time_window} временных шагов, получено {market_data.size(1)}")
        
        # Кодирование признаков
        market_feat = self.market_encoder(market_data)
        static_feat = self.static_encoder(static_data)
        combined = torch.cat([market_feat, static_feat], dim=-1)
        
        # Обработка состояния и прогноз
        processed_state = self.state_processor(combined)
        volume_pct, risk = self.predictor(processed_state)
        
        return volume_pct, risk
    
    def loss_function(self, y_pred, y_true):
        price_loss = self.criterion(y_pred, y_true)
    
        return price_loss