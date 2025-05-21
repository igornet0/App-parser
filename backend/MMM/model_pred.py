from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]
    

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _prob_QK(self, Q, K):
        B, H, L, E = Q.shape
        _, _, S, _ = K.shape

        # Random sampling of keys
        K_sample = K[:, :, torch.randint(S, (self.factor * L,))]  # [B, H, factor*L, d_k]
        Q_K_sample = torch.einsum("bhle,bhse->bhls", Q, K_sample) / math.sqrt(E)
        
        # Measure query importance
        M = Q_K_sample.max(-1)[0] - torch.mean(Q_K_sample, -1)
        return torch.argmax(M, dim=-1)  # Returns [B, H]

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        
        # Projections
        q = self.q_linear(x).view(B, L, H, -1).permute(0, 2, 1, 3)  # [B, H, L, d_k]
        k = self.k_linear(x).view(B, L, H, -1).permute(0, 2, 1, 3)
        v = self.v_linear(x).view(B, L, H, -1).permute(0, 2, 1, 3)
        
        # Get sparse indices
        U = self._prob_QK(q, k)  # [B, H]
        U = U.unsqueeze(-1)  # [B, H, 1]
        
        # Select top queries
        q_reduced = q[
            torch.arange(B)[:, None, None],  # [B, 1, 1]
            torch.arange(H)[None, :, None],  # [1, H, 1]
            U,                              # [B, H, 1]
            :                               # All features
        ]  # [B, H, 1, d_k]
        
        # Attention calculation
        scores = torch.einsum('bhle,bhse->bhls', q_reduced, k) / math.sqrt(self.d_k)  # [B, H, 1, L]
        attn = F.softmax(scores, dim=-1)
        
        # Context aggregation
        context = torch.einsum('bhls,bhse->bhle', attn, v)  # [B, H, 1, d_k]
        
        # We need to expand the context to match original sequence length
        # Since we reduced the queries, we'll distribute the context to all positions
        context = context.expand(-1, -1, L, -1)  # [B, H, L, d_k]
        
        # Combine heads
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.out(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * factor, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout, num_layers, factor):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                factor=factor,
                use_conv=True
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, factor, use_conv):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, factor)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1) if use_conv else None
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, factor, dropout)
        
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        if self.conv:
            conv_out = self.conv(x.transpose(1,2)).transpose(1,2)
            x = x + self.dropout(conv_out)
        
        ff_out = self.ffn(x)
        return self.norm2(x + self.dropout(ff_out))

class MultiScaleDecoder(nn.Module):
    def __init__(self, d_model, pred_len, num_kernels, dropout):
        super().__init__()
        self.pred_len = pred_len
        kernels = [3, 5, 7]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=k//2),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k in kernels[:num_kernels]
        ])
        # Project to d_model dimension
        self.proj = nn.Linear(d_model * num_kernels, d_model)
        
    def forward(self, x):
        # x shape: [B, seq_len, d_model]
        x = x.transpose(1, 2)  # [B, d_model, seq_len]
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        x = torch.cat(outputs, dim=1)  # [B, d_model*num_kernels, seq_len]
        x = x.transpose(1, 2)  # [B, seq_len, d_model*num_kernels]
        x = self.proj(x)  # [B, seq_len, d_model]
        
        # Reduce sequence length to pred_len
        x = x[:, -self.pred_len:, :]
        return x 


class RevIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = x - self.mean
            x = x / (self.stdev + 1e-5)
            return x * self.affine_weight + self.affine_bias
        else:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-5)
            x = x * self.stdev + self.mean
            return x
        
    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True)
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)


class EnhancedTimeSeriesModel(nn.Module):
    def __init__(self, 
                 n_indicators=0,
                 seq_len=30,
                 pred_len=5,
                 d_model=128,
                 n_heads=8,
                 dropout=0.1):
        
        """
        Parameters:
        n_indicators	Количество дополнительных индикаторов (например, RSI, MACD и т.д.).
        seq_len	        Длина входной последовательности (количество временных шагов для анализа).
        pred_len	    Длина прогнозируемой последовательности (количество шагов вперед).
        d_model	        Размерность скрытых представлений (эмбеддингов) в модели.
        n_heads	        Количество голов внимания в механизме ProbSparse Attention.
        dropout	        Вероятность дропаута для регуляризации.

        5 базовых признаков:
        open, high, low, close, volume.

        Ключевые зависимости параметров:
        d_model должен делиться на n_heads (для работы многоголового внимания).
        num_kernels не может превышать 3 (так как определены ядра размеров 3, 5, 7).
        time_data должна содержать 7 признаков (соответствующих году, месяцу и т.д.).
        """
        super().__init__()
        self.pred_len = pred_len
        self.base_features = 5  # open, high, low, close, volume
        self.n_indicators = n_indicators
        self.total_features = self.base_features + n_indicators

        # Time feature processing
        self.time_embed = nn.ModuleDict({
            'month': nn.Embedding(12, 8),
            'day': nn.Embedding(31, 8),
            'weekday': nn.Embedding(7, 8),
            'hour': nn.Embedding(24, 8),
            'minute': nn.Embedding(60, 8),
            'time_cos': nn.Linear(1, 8),
            'time_sin': nn.Linear(1, 8)
        })
        self.time_proj = nn.Linear(8*7, d_model)

        # Data normalization
        self.revin = RevIN(num_features=self.total_features)

        # Main embeddings
        self.value_embed = nn.Linear(self.total_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len)

        # Transformer components
        self.encoder = HierarchicalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            num_layers=4,
            factor=4
        )
        
        # Multi-scale decoding
        self.decoder = MultiScaleDecoder(
            d_model=d_model,
            pred_len=pred_len,
            num_kernels=3,
            dropout=dropout
        )

        # Output projections
        self.price_proj = nn.Linear(d_model, pred_len)  # Predict close price
        self.volatility_proj = nn.Linear(d_model, pred_len)  # Predict volatility

    def forward(self, x, time_features):
        # x: [batch, seq_len, 5 + n_indicators]
        # datetimes: [batch, seq_len] (datetime strings)
        
        # Normalize all features
        x = self.revin(x, 'norm')

        # Cyclic encoding for time and day of year
        hour_min = time_features[..., 3] + time_features[..., 4]/60
        time_angle = 2 * math.pi * hour_min / 24
        time_cos = torch.cos(time_angle).unsqueeze(-1)
        time_sin = torch.sin(time_angle).unsqueeze(-1)
        
        # day_angle = 2 * math.pi * time_features[..., 5] / 365
        # day_cos = torch.cos(day_angle).unsqueeze(-1)
        # day_sin = torch.sin(day_angle).unsqueeze(-1)
        
        # Embeddings
        embeddings = []
        for key, layer in self.time_embed.items():
            if key in ['month', 'day', 'weekday', 'hour', 'minute']:
                idx = ['month', 'day', 'weekday', 'hour', 'minute'].index(key)
                inputs = time_features[..., idx].long()
                inputs = self._normalize_time(key, inputs)
                embeddings.append(layer(inputs))
            elif key == 'time_cos':
                embeddings.append(layer(time_cos))
            elif key == 'time_sin':
                embeddings.append(layer(time_sin))
        
        embeddings = torch.cat(embeddings, dim=-1)
        time_emb = self.time_proj(embeddings)
        
        # Process time features
        # time_emb = self._process_datetime(datetimes)
        
        # Combine embeddings
        x_emb = self.value_embed(x) + time_emb
        x_emb = self.pos_enc(x_emb)
        
        # Encode-decode
        encoded = self.encoder(x_emb)
        decoded = self.decoder(encoded)
        
        # Predictions
        price_pred = self.price_proj(decoded)
        volatility_pred = self.volatility_proj(decoded)
        
        # Denormalize only close price
        close_mean = self.revin.mean[..., 3:4]
        close_stdev = self.revin.stdev[..., 3:4]
        
        price_pred = price_pred * close_stdev + close_mean
        volatility_pred = volatility_pred * close_stdev  # Volatility in same scale
        
        return torch.cat([price_pred, volatility_pred], dim=-1)
    
    def _normalize_time(self, key, inputs):
        if key == 'month': return inputs - 1
        if key == 'day': return inputs - 1
        return inputs
    
    def _get_time_index(self, key): 
        return {
            'year': 0, 'month': 1, 'day': 2, 'hour': 3,
            'minute': 4, 'second': 5, 'weekday': 6
        }[key]

    def _normalize_time_inputs(self, key, inputs):
        if key == 'year': 
            return inputs - 2020  # Для годов 2020-2029 → индексы 0-9
        elif key == 'month': 
            return torch.clamp(inputs - 1, 0, 11)  # 1-12 → 0-11
        elif key == 'day': 
            return torch.clamp(inputs - 1, 0, 30)  # 1-31 → 0-30
        elif key == 'hour': 
            return torch.clamp(inputs, 0, 23)
        elif key == 'minute' or key == 'second':
            return torch.clamp(inputs, 0, 59)
        elif key == 'weekday': 
            return torch.clamp(inputs, 0, 6)
        return inputs

if __name__ == "__main__":
    # Генерация реалистичных временных данных
    time_data = torch.stack([
        torch.randint(2020, 2030, (32, 30)),    # Year
        torch.randint(1, 13, (32, 30)),         # Month
        torch.randint(1, 32, (32, 30)),         # Day
        torch.randint(0, 24, (32, 30)),         # Hour
        torch.randint(0, 60, (32, 30)),         # Minute
        torch.randint(0, 60, (32, 30)),         # Second
        torch.randint(0, 7, (32, 30)),          # Weekday
    ], dim=-1).float()
    
    model = EnhancedTimeSeriesModel()
    main_data = torch.randn(32, 30, 5)
    output = model(main_data, time_data)
    print(f"Output shape: {output.shape}")