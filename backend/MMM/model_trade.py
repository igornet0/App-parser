import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 1. MLP (Feedforward)
class OHLCV_MLP(nn.Module):
    def __init__(self, input_size, seq_len=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_size*seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# 2. LSTM
class OHLCV_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 3)
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])

# 3. Temporal CNN
class OHLCV_TCNN(nn.Module):
    def __init__(self, input_size, seq_len=30):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, input_size))
        )
        self.fc = nn.Linear(32*input_size, 3)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len, features]
        x = self.conv(x)
        return self.fc(x.flatten(1))

# 4. Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Рассчитываем div_term для всех возможных индексов
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term[:d_model//2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :self.d_model]

class OHLCV_Transformer(nn.Module):
    def __init__(self, input_size, nhead=4, num_layers=3):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_size)
        encoder_layers = TransformerEncoderLayer(input_size, nhead, 256)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(input_size, 3)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

# 5. CNN-LSTM Hybrid
class OHLCV_CNN_LSTM(nn.Module):
    def __init__(self, input_size, conv_features=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(conv_features, 64, batch_first=True)
        self.fc = nn.Linear(64, 3)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# 6. TabTransformer
class TabTransformer(nn.Module):
    def __init__(self, num_features, cat_features, cat_dims, dim=32):
        super().__init__()
        # Embedding layers с фиксированной выходной размерностью
        self.embeds = nn.ModuleList([
            nn.Embedding(cat_dim, dim) for cat_dim in cat_dims
        ])
        
        # Проекция объединенных признаков в dimension трансформера
        input_dim = num_features + len(cat_dims) * dim
        self.input_proj = nn.Linear(input_dim, dim)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Финальный MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x_num, x_cat):
        # Эмбеддинги категориальных признаков
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        
        # Конкатенация и проекция
        x = torch.cat([x_num] + cat_embeds, dim=1)
        x = self.input_proj(x)
        
        # Трансформер и MLP
        x = self.encoder(x)
        return self.mlp(x.mean(dim=1))