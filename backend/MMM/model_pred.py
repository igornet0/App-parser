import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(1, max_len, d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#         self.d_model = d_model

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, seq_len, embedding_dim]
#         """
#         return x * math.sqrt(self.d_model) + self.pe[:, :x.size(1)]


# class ProbSparseAttention(nn.Module):
#     def __init__(self, d_model, n_heads, factor=5):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_k = d_model // n_heads
#         self.factor = factor
        
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.out = nn.Linear(d_model, d_model)

#     def _prob_QK(self, Q, K):
#         B, H, L, E = Q.shape
#         S = K.size(2)  # Sequence length dimension
        
#         # Adjust sampling to avoid exceeding dimension size
#         u = self.factor * int(math.log(L))  # Sample factor * log(L) keys
#         u = min(u, S)  # Ensure we don't sample more than available
#         K_sample_indices = torch.randperm(S)[:u]
#         K_sample = K[:, :, K_sample_indices, :]  # [B, H, u, d_k]
        
#         # Compute QK scores with sampled keys
#         Q_K_sample = torch.einsum('bhld,bhud->bhlu', Q, K_sample)  # [B, H, L, u]
        
#         # Compute importance metric M
#         M = Q_K_sample.std(dim=-1) / (Q_K_sample.mean(dim=-1).abs() + 1e-5)  # [B, H, L]
        
#         # Select top L//2 queries per head
#         topk_indices = torch.topk(M, k=L//2, dim=-1)[1]  # [B, H, L//2]
#         return topk_indices

#     def forward(self, x):
#         B, L, _ = x.shape
#         H = self.n_heads
        
#         # Project and reshape queries, keys, values
#         q = self.q_linear(x).view(B, L, H, self.d_k).permute(0, 2, 1, 3)  # [B, H, L, d_k]
#         k = self.k_linear(x).view(B, L, H, self.d_k).permute(0, 2, 1, 3)  # [B, H, L, d_k]
#         v = self.v_linear(x).view(B, L, H, self.d_k).permute(0, 2, 1, 3)   # [B, H, L, d_k]
        
#         # Get indices of top queries to keep
#         U = self._prob_QK(q, k)  # [B, H, L//2]
        
#         # Gather the selected queries
#         q_reduced = torch.gather(
#             q, 
#             dim=2, 
#             index=U.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
#         )  # [B, H, L//2, d_k]
        
#         # Compute attention scores with ALL keys
#         scores = torch.einsum('bhqd,bhld->bhql', q_reduced, k) / math.sqrt(self.d_k)
#         attn = F.softmax(scores, dim=-1)  # [B, H, L//2, L]
        
#         # Aggregate context using attention weights
#         context = torch.einsum('bhql,bhld->bhqd', attn, v)  # [B, H, L//2, d_k]
        
#         # Scatter reduced context back to full sequence length
#         context_full = torch.zeros(
#             B, H, L, self.d_k, 
#             device=x.device,
#             dtype=context.dtype  # Match dtype with source tensor
#         )
#         context_full.scatter_(
#             dim=2,
#             index=U.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
#             src=context
#         )
        
#         # Combine heads and project output
#         output = context_full.permute(0, 2, 1, 3).reshape(B, L, -1)
#         return self.out(output)


# class FeedForward(nn.Module):
#     def __init__(self, d_model, factor=4, dropout=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, d_model * factor),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * factor, d_model),
#             nn.Dropout(dropout)
#         )
        
#     def forward(self, x):
#         return self.net(x)


# class HierarchicalTransformerEncoder(nn.Module):
#     def __init__(self, d_model, n_heads, dropout, num_layers, factor):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             TransformerBlock(
#                 d_model=d_model,
#                 n_heads=n_heads,
#                 dropout=dropout,
#                 factor=factor,
#                 use_conv=True
#             ) for _ in range(num_layers)
#         ])
        
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, n_heads, dropout, factor, use_conv):
#         super().__init__()
#         self.attention = ProbSparseAttention(d_model, n_heads, factor)
#         self.conv = nn.Conv1d(d_model, d_model, 3, padding=1) if use_conv else None
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.ffn = FeedForward(d_model, factor, dropout)
        
#     def forward(self, x):
#         attn_out = self.attention(x)
#         x = self.norm1(x + self.dropout(attn_out))
        
#         if self.conv:
#             conv_out = self.conv(x.transpose(1,2)).transpose(1,2)
#             x = x + self.dropout(conv_out)
        
#         ff_out = self.ffn(x)
#         return self.norm2(x + self.dropout(ff_out))

# class MultiScaleDecoder(nn.Module):
#     def __init__(self, d_model, pred_len, num_kernels, dropout):
#         super().__init__()
#         self.pred_len = pred_len
#         kernels = [3, 5, 7]
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(d_model, d_model, k, padding=k//2),
#                 nn.GELU(),
#                 nn.Dropout(dropout)
#             ) for k in kernels[:num_kernels]
#         ])
#         # Project to d_model dimension
#         self.proj = nn.Linear(d_model * num_kernels, d_model)
        
#     def forward(self, x):
#         # x shape: [B, seq_len, d_model]
#         x = x.transpose(1, 2)  # [B, d_model, seq_len]
#         outputs = []
#         for conv in self.convs:
#             outputs.append(conv(x))
#         x = torch.cat(outputs, dim=1)  # [B, d_model*num_kernels, seq_len]
#         x = x.transpose(1, 2)  # [B, seq_len, d_model*num_kernels]
#         x = self.proj(x)  # [B, seq_len, d_model]
        
#         # Reduce sequence length to pred_len
#         x = x[:, -self.pred_len:, :]
#         return x 


# class RevIN(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.num_features = num_features
#         self.affine_weight = nn.Parameter(torch.ones(num_features))
#         self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
#     def forward(self, x, mode):
#         if mode == 'norm':
#             self._get_statistics(x[..., :self.num_features])  
#             x_norm = (x[..., :self.num_features] - self.mean) / (self.stdev + 1e-5)
#             x_norm = x_norm * self.affine_weight + self.affine_bias  # Аффинное преобразование
#             return torch.cat([x_norm, x[..., self.num_features:]], dim=-1)
#         else:
#             x_denorm = x.clone()
#             x_denorm[..., :self.num_features] = (x_denorm[..., :self.num_features] - self.affine_bias) / (self.affine_weight + 1e-5)
#             x_denorm[..., :self.num_features] = x_denorm[..., :self.num_features] * self.stdev + self.mean
#             return x_denorm
        
#     def _get_statistics(self, x):
#         self.mean = torch.mean(x, dim=1, keepdim=True)
#         self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)


# class PricePredictorModel(nn.Module):

#     criterion = torch.nn.MSELoss()

#     def __init__(self, 
#                  n_indicators=0,
#                  input_features=['open', 'high', 'low', 'close', 'volume'],
#                  seq_len=30,
#                  pred_len=5,
#                  d_model=128,
#                  n_heads=8,
#                  dropout=0.1):
        
#         """
#         Parameters:
#         n_indicators	Количество дополнительных индикаторов (например, RSI, MACD и т.д.).
#         input_features  
#         seq_len	        Длина входной последовательности (количество временных шагов для анализа).
#         pred_len	    Длина прогнозируемой последовательности (количество шагов вперед).
#         d_model	        Размерность скрытых представлений (эмбеддингов) в модели.
#         n_heads	        Количество голов внимания в механизме ProbSparse Attention.
#         dropout	        Вероятность дропаута для регуляризации.

#         5 базовых признаков:
#         open, high, low, close, volume.

#         Ключевые зависимости параметров:
#         d_model должен делиться на n_heads (для работы многоголового внимания).
#         num_kernels не может превышать 3 (так как определены ядра размеров 3, 5, 7).
#         time_data должна содержать 7 признаков (соответствующих году, месяцу и т.д.).
#         """
#         super().__init__()
#         self.pred_len = pred_len
#         self.input_features = input_features
#         self.n_indicators = n_indicators
#         self.total_features = len(input_features) + n_indicators

#         # Time feature processing
#         self.time_embed = nn.ModuleDict({
#             'month': nn.Embedding(12, 8),
#             'day': nn.Embedding(31, 8),
#             'weekday': nn.Embedding(7, 8),
#             'hour': nn.Embedding(24, 8),
#             'minute': nn.Embedding(60, 8),
#             'time_cos': nn.Linear(1, 8),
#             'time_sin': nn.Linear(1, 8)
#         })
#         self.time_proj = nn.Linear(8*7, d_model)

#         # Data normalization
#         self.revin = RevIN(num_features=self.total_features)

#         # Main embeddings
#         self.value_embed = nn.Linear(self.total_features, d_model)
#         self.pos_enc = PositionalEncoding(d_model, seq_len)

#         # Transformer components
#         self.encoder = HierarchicalTransformerEncoder(
#             d_model=d_model,
#             n_heads=n_heads,
#             dropout=dropout,
#             num_layers=4,
#             factor=4
#         )
        
#         # Multi-scale decoding
#         self.decoder = MultiScaleDecoder(
#             d_model=d_model,
#             pred_len=pred_len,
#             num_kernels=3,
#             dropout=dropout
#         )

#         # Output projections
#         self.price_proj = nn.Linear(d_model, 1)  # Predict single value per timestep

#     def forward(self, x, time_features):
#         # x: [batch, seq_len, 5 + n_indicators]
#         # datetimes: [batch, seq_len] (datetime strings)
        
#         # Normalize all features
#         x = self.revin(x, 'norm')

#         # Cyclic encoding for time and day of year
#         hour_min = time_features[..., 3] + time_features[..., 4]/60
#         time_angle = 2 * math.pi * hour_min / 24
#         time_cos = torch.cos(time_angle).unsqueeze(-1)
#         time_sin = torch.sin(time_angle).unsqueeze(-1)

#         # Embeddings
#         embeddings = []
#         for key, layer in self.time_embed.items():
#             if key in ['month', 'day', 'weekday', 'hour', 'minute']:
#                 idx = ['month', 'day', 'weekday', 'hour', 'minute'].index(key)
#                 inputs = time_features[..., idx].long()
#                 inputs = self._normalize_time(key, inputs)
#                 embeddings.append(layer(inputs))
#             elif key == 'time_cos':
#                 embeddings.append(layer(time_cos))
#             elif key == 'time_sin':
#                 embeddings.append(layer(time_sin))
        
#         # Combine all embeddings
#         time_emb = self.time_proj(torch.cat(embeddings, dim=-1))
#         value_emb = self.value_embed(x)
        
#         # Sum all embeddings
#         x_emb = value_emb + time_emb
#         x_emb = self.pos_enc(x_emb)
        
#         # Encode-decode
#         encoded = self.encoder(x_emb)
#         decoded = self.decoder(encoded)  # [B, pred_len, d_model]
        
#         # Predict price
#         price_pred = self.price_proj(decoded).squeeze(-1)  # [B, pred_len]
        
#         # Denormalize using close price statistics
#         close_mean = self.revin.mean[..., 3:4]
#         close_stdev = self.revin.stdev[..., 3:4]
        
#         return price_pred * close_stdev.squeeze(-1) + close_mean.squeeze(-1)
    
#     def loss_function(self, y_pred, y_true):
#         price_loss = self.criterion(y_pred, y_true)
    
#         return price_loss
    
#     def _normalize_time(self, key, inputs):
#         if key == 'month':
#             return inputs - 1  # Convert 1-12 to 0-11
#         elif key == 'day':
#             return inputs - 1  # Convert 1-31 to 0-30
#         elif key == 'weekday':
#             return torch.clamp(inputs, 0, 6)  # Clamp 7 to 6
#         else:
#             return inputs
    
#     def _get_time_index(self, key): 
#         return {
#             'year': 0, 'month': 1, 'day': 2, 'hour': 3,
#             'minute': 4, 'second': 5, 'weekday': 6
#         }[key]

#     def _normalize_time_inputs(self, key, inputs):
#         if key == 'year': 
#             return inputs - 2020  # Для годов 2020-2029 → индексы 0-9
#         elif key == 'month': 
#             return torch.clamp(inputs - 1, 0, 11)  # 1-12 → 0-11
#         elif key == 'day': 
#             return torch.clamp(inputs - 1, 0, 30)  # 1-31 → 0-30
#         elif key == 'hour': 
#             return torch.clamp(inputs, 0, 23)
#         elif key == 'minute' or key == 'second':
#             return torch.clamp(inputs, 0, 59)
#         elif key == 'weekday': 
#             return torch.clamp(inputs, 0, 6)
#         return inputs

class PricePredictorModel(nn.Module):

    criterion = nn.HuberLoss(delta=0.5)

    def __init__(self, pred_len, seq_len, num_features=5, 
                 n_heads=8, d_model=128, 
                 emb_month_size=8, 
                 emb_weekday_size=4, 
                 lstm_hidden=256, 
                 num_layers=2, 
                 dropout=0.2):
        """
        Args:
            pred_len: Количество шагов для предсказания
            seq_len: Длина входной последовательности
            num_features: Количество признаков в X (5)
            time_features: Количество временных признаков (5)
            emb_month_size: Размер эмбеддинга для месяца
            emb_weekday_size: Размер эмбеддинга для дня недели
            lstm_hidden: Размер скрытого состояния LSTM
            num_layers: Количество слоев LSTM
            dropout: Процент дропаута
        """
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        
        # Временные эмбеддинги
        self.emb_month = nn.Embedding(12, emb_month_size)
        self.emb_weekday = nn.Embedding(8, emb_weekday_size)
        
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
        
        # Блок предсказания
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len)
        )
        
        # Регуляризация
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features + 11)

    def forward(self, x, time):
        """
        x: [batch_size, seq_len, num_features-5] - основные фичи
        time: [batch_size, seq_len, 5] - [месяц, день, час, минута, день_недели]
        """
        
        # 1. Обработка временных признаков
        month = time[:, :, 0].long() - 1  # [batch_size, seq_len] (0-11)
        weekday = time[:, :, 4].long()  # [batch_size, seq_len] (0-7)
        day = time[:, :, 1].unsqueeze(-1)  # [batch_size, seq_len, 1]
        hour = time[:, :, 2].unsqueeze(-1)  # [batch_size, seq_len, 1]
        minute = time[:, :, 3].unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Цикличные кодирования
        day_sin = torch.sin(day * (2 * math.pi / 31))
        day_cos = torch.cos(day * (2 * math.pi / 31))
        hour_sin = torch.sin(hour * (2 * math.pi / 24))
        hour_cos = torch.cos(hour * (2 * math.pi / 24))
        minute_sin = torch.sin(minute * (2 * math.pi / 60))
        minute_cos = torch.cos(minute * (2 * math.pi / 60))
        
        # Эмбеддинги
        month_emb = self.emb_month(month)  # [batch_size, seq_len, emb_month_size]
        weekday_emb = self.emb_weekday(weekday)  # [batch_size, seq_len, emb_weekday_size]
        
        # Комбинирование временных фич
        time_features = torch.cat([
            month_emb,
            weekday_emb,
            day_sin, day_cos,
            hour_sin, hour_cos,
            minute_sin, minute_cos
        ], dim=-1)

        # print("time_features: ", time_features.shape)
        
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
        
        # 6. Предсказание
        predictions = self.fc(context_vector)  # [batch_size, pred_len]
        
        return predictions
    
    def loss_function(self, y_pred, y_true):
        price_loss = self.criterion(y_pred, y_true)
    
        return price_loss

if __name__ == "__main__":
    # Генерация реалистичных временных данных
    time_data = torch.stack([
        torch.randint(1, 13, (32, 30)),         # Month
        torch.randint(1, 32, (32, 30)),         # Day
        torch.randint(0, 24, (32, 30)),         # Hour
        torch.randint(0, 60, (32, 30)),         # Minute
        torch.randint(0, 8, (32, 30)),          # Weekday
    ], dim=-1).float()
    
    model = PricePredictorModel(5, 30)
    main_data = torch.randn(32, 30, 5)
    
    print(f"Input shape: {main_data.shape}, Time shape: {time_data.shape}")
    print(f"Input sample: {main_data[0]}")  # Печать первого примера из батча
    print(f"Time sample: {time_data[0]}")  # Печать первого временного примера из батча
    print(f"Input values: {main_data[0].tolist()}")  # Печать значений первого примера
    print(f"Time values: {time_data[0].tolist()}")  # Печать значений первого временного примера
    print(f"Input mean: {main_data.mean().item()}")  # Печать среднего значения входных данных
    print(f"Input std: {main_data.std().item()}")  # Печать стандартного отклонения входных данных
    print(f"Input min: {main_data.min().item()}")  # Печать минимального значения входных данных
    print(f"Input max: {main_data.max().item()}")  # Печать максимального значения входных данных
    print(f"Input range: {main_data.min().item()} - {main_data.max().item()}")  # Печать диапазона входных данных
    
    
    output = model(main_data, time_data)


    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}")  # Печать первого примера из батча
    print(f"Output values: {output[0].tolist()}")  # Печать значений первого примера
    print(f"Output mean: {output.mean().item()}")  # Печать среднего значения выходных данных
    print(f"Output std: {output.std().item()}")  # Печать стандартного отклонения выходных данных
    print(f"Output min: {output.min().item()}")  # Печать минимального значения выходных данных
    print(f"Output max: {output.max().item()}")  # Печать максимального значения выходных данных
    print(f"Output range: {output.min().item()} - {output.max().item()}")  # Печать диапазона выходных данных
    print(f"Output first 5 values: {output[0][:5].tolist()}")  # Печать первых 5 значений первого примера
    print(f"Output last 5 values: {output[0][-5:].tolist()}")  # Печать последних 5 значений первого примера
    print(f"Output first 5 values (rounded): {[round(val, 2) for val in output[0][:5].tolist()]}")  # Печать первых 5 значений с округлением
    print(f"Output last 5 values (rounded): {[round(val, 2) for val in output[0][-5:].tolist()]}")  # Печать последних 5 значений с округлением