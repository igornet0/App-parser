import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EmbeddingDatetime(nn.Module):
    def __init__(self, emb_month_size, emb_weekday_size):
        super().__init__()
        # Временные эмбеддинги
        self.emb_month = nn.Embedding(12, emb_month_size)
        self.emb_weekday = nn.Embedding(8, emb_weekday_size)
        
    def forward(self, time):
        # 1. Обработка временных признаков
        month = time[:, :, 0].long() - 1  # [batch_size, seq_len] (0-11)
        day = time[:, :, 1].unsqueeze(-1)  # [batch_size, seq_len, 1]
        hour = time[:, :, 2].unsqueeze(-1)  # [batch_size, seq_len, 1]
        minute = time[:, :, 3].unsqueeze(-1)  # [batch_size, seq_len, 1]
        weekday = time[:, :, 4].long()  # [batch_size, seq_len] (0-7)
        
        # Цикличные кодирования
        day_sin = torch.sin(day * (2 * math.pi / 31))
        day_cos = torch.cos(day * (2 * math.pi / 31))
        hour_sin = torch.sin(hour * (2 * math.pi / 24))
        hour_cos = torch.cos(hour * (2 * math.pi / 24))
        minute_sin = torch.sin(minute * (2 * math.pi / 60))
        minute_cos = torch.cos(minute * (2 * math.pi / 60))
        
        # Эмбеддинги
        month_emb = self.emb_month(month)  # [batch_size, seq_len, emb_month_size]8
        weekday_emb = self.emb_weekday(weekday)  # [batch_size, seq_len, emb_weekday_size]4
        
        return torch.cat([
            month_emb,
            weekday_emb,
            day_sin, day_cos,
            hour_sin, hour_cos,
            minute_sin, minute_cos
        ], dim=-1)


class LTSMTimeFrame(nn.Module):
    def __init__(self, emb_month_size=8, emb_weekday_size=4, num_features=6, lstm_hidden=256, num_layers=2, n_heads=4, dropout=0.2):
        super().__init__()

        self.EmbeddingLayer = EmbeddingDatetime(emb_month_size, emb_weekday_size)

        # Параметры для временных преобразований
        self.time_linear = nn.Linear(
            emb_month_size + emb_weekday_size + 6,  # month + weekday + hour + minute + day
            16
        )
        
        # Основной LSTM блок
        self.lstm = nn.LSTM(
            input_size=num_features + 11,  # Основные фичи + временные фичи
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Механизм внимания
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(num_features + 11)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time):
        
        time_features = self.EmbeddingLayer(time)

        time_features = F.gelu(self.time_linear(time_features))
        
        # 2. Объединение с основными фичами
        combined = torch.cat([x, time_features], dim=-1)  # [batch_size, seq_len, num_features+16]
    
        combined = self.layer_norm(combined)
        
        # 3. LSTM обработка
        lstm_out, _ = self.lstm(combined)  # [batch_size, seq_len, lstm_hidden]
        lstm_out = self.dropout(lstm_out)
        
        # 4. Механизм внимания
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out  # self-attention
        )  # [batch_size, seq_len, lstm_hidden]
        
        # 5. Агрегация по временной оси (с сохранением временной структуры)
        context_vector = attn_out.mean(dim=1)  # [batch_size, lstm_hidden]
        
        return context_vector