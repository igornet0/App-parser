import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class MarketFeatureExtractor(nn.Module):
    """Извлекает временные паттерны из исторических данных"""
    def __init__(self, input_channels=5, features=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, features, batch_first=True)
        self.attention = nn.MultiheadAttention(features, num_heads=4, batch_first=True)
        
    def forward(self, x):
        # x: (batch, timesteps, features) -> (batch, features, timesteps)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, timesteps, features)
        
        # Временные зависимости
        lstm_out, _ = self.lstm(x)
        
        # Внимание к ключевым периодам
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out[:, -1, :]  # Последний контекст


class RiskAwareActor(nn.Module):
    """Генератор действий с учётом риска"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Выходы для непрерывных действий
        self.volume_mu = nn.Linear(hidden_dim, 1)
        self.volume_sigma = nn.Linear(hidden_dim, 1)
        
        # Выходы для дискретных действий
        self.order_type = nn.Linear(hidden_dim, 2)  # Лимитный/рыночный
        self.position_side = nn.Linear(hidden_dim, 2)  # Long/Short

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Распределение для объема
        volume_mu = F.softplus(self.volume_mu(x))  # Всегда положительный
        volume_sigma = F.softplus(self.volume_sigma(x)) + 1e-5
        
        # Распределение для типа ордера
        order_type_logits = self.order_type(x)
        
        # Распределение для позиции
        position_logits = self.position_side(x)
        
        return (volume_mu, volume_sigma), order_type_logits, position_logits


class ValueCritic(nn.Module):
    """Критик для оценки стоимости состояний"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v_out(x)


class PositionTracker(nn.Module):
    """Отслеживает состояние позиции во времени"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(5, hidden_dim, batch_first=True)  # Вход: цена, объем, время
        self.fc = nn.Linear(hidden_dim, 3)  # Вывод: риск, прибыль, время до закрытия
        
    def forward(self, position_history):
        _, (hidden, _) = self.lstm(position_history)
        return self.fc(hidden.squeeze(0))


class RiskAwareSACNetwork(nn.Module):
    def __init__(
        self,
        historical_features=5,  # 5 признаков
        time_window=60, # Временное окно
        static_features=4,  # 4 статических признака
        hidden_dim=128, # Размер скрытого слоя
        risk_max=0.05  # RISK_MAX = 5%
    ):
        super().__init__()
        self.time_window = time_window  
        self.risk_max = risk_max
        
        # Основные компоненты из предыдущей архитектуры
        self.market_encoder = MarketFeatureExtractor(historical_features, hidden_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU()
        )
        
        # Трекер позиции
        self.position_tracker = PositionTracker(hidden_dim)
        
        # Актор-критик с расширенным входом (состояние + история позиции)
        combined_dim = 2 * hidden_dim + 3  # +3 от трекера позиции
        self.actor = RiskAwareActor(combined_dim, 1, 2 * hidden_dim)
        self.critic = ValueCritic(combined_dim, 2 * hidden_dim)
        
        ...
        
    def forward(self, market_data, static_data, position_history=None):
        if market_data.size(1) != self.time_window:
            raise ValueError(f"Ожидается временное окно {self.time_window}, получено {market_data.size(1)}")
        
        # Базовое кодирование состояния
        market_feat = self.market_encoder(market_data)
        static_feat = self.static_encoder(static_data)
        state = torch.cat([market_feat, static_feat], dim=-1)
        encoded_state = self.state_encoder(state)
        
        # Если есть история позиции - добавляем
        if position_history is not None:
            position_features = self.position_tracker(position_history)
            full_state = torch.cat([encoded_state, position_features], dim=-1)
        else:
            full_state = encoded_state
        
        # Прогнозирование компонентов
        (vol_mu, vol_sigma), order_logits, pos_logits = self.actor(full_state)
        value = self.critic(full_state)
        profit_pred = self.profit_predictor(full_state)
        
        return (vol_mu, vol_sigma), order_logits, pos_logits, value, profit_pred
    
    def evaluate_position(self, position_history):
        """Оценивает риск и прибыльность текущей позиции"""
        with torch.no_grad():
            features = self.position_tracker(position_history)
            risk_level = torch.sigmoid(features[:, 0])
            profit_potential = torch.tanh(features[:, 1])
            time_to_close = torch.sigmoid(features[:, 2])
            
            # Проверка соблюдения риск-лимитов
            risk_ok = (risk_level < self.risk_max).float()
            
            return {
                'risk': risk_level.item(),
                'profit': profit_potential.item(),
                'time_to_close': time_to_close.item(),
                'risk_ok': risk_ok.item()
            }
        
class TradingPolicyNetwork(nn.Module):
    
    def act(self, market_data, static_data, deterministic=False):
        with torch.no_grad():
            _, order_logits, pos_logits, _, _ = self(market_data, static_data)
            
            # Выбор дискретных действий
            order_type_dist = Categorical(logits=order_logits)
            position_dist = Categorical(logits=pos_logits)
            
            if deterministic:
                order_type = order_type_dist.probs.argmax(-1)
                position = position_dist.probs.argmax(-1)
            else:
                order_type = order_type_dist.sample()
                position = position_dist.sample()
                
            return order_type.item(), position.item()

    def generate_ladder(self, market_data, static_data, levels=3):
        """Генерация лесенки ордеров"""
        (vol_mu, vol_sigma), _, _, _, risk_level = self(market_data, static_data)
        base_volume = Normal(vol_mu, vol_sigma).sample()
        risk_factor = 1.0 - risk_level  # Чем выше риск, тем меньше объем
        
        orders = []
        for i in range(levels):
            # Распределение объема по уровням
            volume = base_volume * (0.4 if i == 0 else 0.6/(levels-1)) * risk_factor
            
            # Расчет цены на основе волатильности
            volatility = market_data[:, -1, 3] - market_data[:, -1, 2]  # High - Low
            price_offset = (i * 0.005 * volatility)  # Шаг 0.5% от волатильности
            
            orders.append({
                'volume': volume.item(),
                'price_offset': price_offset.item()
            })
        return orders

