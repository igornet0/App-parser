import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_ltsm import LTSMTimeFrame

class TradingModel(nn.Module):

    def __init__(self, 
                 seq_len=30,
                 input_size=13, 
                 pred_len=6, 
                 lstm_hidden=256,
                 hidden_size=128,
                 num_layers=2,
                 emb_month_size=8,
                 emb_weekday_size=4,
                 n_heads=4,
                 dropout=0.3):
        super(TradingModel, self).__init__()

        # Timeframe
        self.seq_len = seq_len
        self.pred_len = pred_len

        # total_features = input_size + pred_len + volume, risk score
        self.total_features = input_size + pred_len
        self.input_size = input_size

        # print("total_features=", total_features)
        # print("input_size=", input_size)

        # сновной LSTM для обработки временных рядов
        self.encode_lstm = LTSMTimeFrame(
            emb_month_size=emb_month_size,
            emb_weekday_size=emb_weekday_size,
            num_features=input_size,
            lstm_hidden=lstm_hidden,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout
        ) # -> 256 (hidden_size)
        
        # Декодер для шагов предсказания
        self.decoder_lstm = nn.LSTM(
            input_size=lstm_hidden + 1,  # +1 для признака шага предсказания
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Механизм внимания
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.context_proj = nn.Linear(lstm_hidden, hidden_size)
        self.step_embedding = nn.Embedding(pred_len, lstm_hidden)
        
        # Выходные слои
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, 3)
        
        # Вспомогательные слои
        # self.layer_norm = nn.LayerNorm(input_size - 5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_pred, time_data):

        # x [B, seq_len, input_size]
        # x_pred [B, pred_len]
        # time_data [B, seq_len, 5]

        # Нормализация истории
        # x_norm = self.layer_norm(x)
        # x_norm = self.dropout(x_norm)
        
        # Кодируем историю
        # context = self.encode_lstm(x, time_data)  # [B, hidden_size]
        
        # Подготовка для декодера
        batch_size = x.size(0)
        hidden = None
        all_outputs = []
        
        # Создаем последовательность шагов [0, 1, ..., pred_len-1]
        steps = torch.arange(self.pred_len, device=x.device).repeat(batch_size, 1)  # [B, pred_len]
        
        context = self.encode_lstm(x, time_data)  # [B, 256]
        context_proj = self.context_proj(context)  # [B, 128]
        context_3d = context_proj.unsqueeze(1)    # [B, 1, 128]

        # Итерация по каждому шагу предсказания
        for step in range(self.pred_len):
            # Выбираем признак для текущего шага
            step_feature = x_pred[:, step].unsqueeze(1)  # [B, 1]
            
            # Эмбеддинг текущего шага
            step_emb = self.step_embedding(steps[:, step])  # [B, hidden_size]

            modified_context = context + step_emb  # Add temporal information
            
            # Контекст из истории (используем последнее состояние)
            # context = encoder_out[:, -1, :]  # [B, hidden_size]
            
            # Вход декодера = контекст + признак шага
            decoder_input = torch.cat([
                modified_context.unsqueeze(1), 
                step_feature.unsqueeze(1)
            ], dim=-1)

            # decoder_input = torch.cat([
            #     context.unsqueeze(1),       # [B, 1, 256]
            #     step_feature.unsqueeze(1)   # [B, 1, 1]
            # ], dim=-1) # [B, 1, hidden_size + 1]
            context_proj = self.context_proj(context)  # [B, 128]
            context_3d = context_proj.unsqueeze(1)  # Добавляем seq_len=1
            
            # Обработка в LSTM декодере
            decoder_out, hidden = self.decoder_lstm(decoder_input, hidden)
            
            # Внимание между выходом декодера и историей
            attn_out, _ = self.attention(
                query=decoder_out,  # [B, 1, 128]
                key=context_3d,     # [B, 1, 128]
                value=context_3d,   # [B, 1, 128]
            )    # [B, 1, hidden_size]
            
            # Объединение информации
            combined = torch.cat([decoder_out, attn_out], dim=-1)  # [B, 1, hidden_size * 2]
            
            # Полносвязные слои
            x_fc = F.relu(self.fc1(combined))
            x_fc = self.dropout(x_fc)
            x_fc = F.relu(self.fc2(x_fc))
            x_fc = self.dropout(x_fc)
            logits = self.output(x_fc)  # [B, 1, 3]
            
            all_outputs.append(logits)
        
        # Собираем все выходы
        output_tensor = torch.cat(all_outputs, dim=1)  # [B, pred_len, 3]
        
        # Преобразование в вероятности 0-100%
        probs = F.softmax(output_tensor, dim=-1) * 100
        
        return probs  # Размер: [B, pred_len, 3]
    
    def loss_function(self, criterion, y_pred, y_true):    
        return criterion(y_pred, y_true)