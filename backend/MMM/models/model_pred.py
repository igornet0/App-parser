import torch
import torch.nn as nn

from .model_ltsm import LTSMTimeFrame

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
        

        self.LSTMTimeFrameLare = LTSMTimeFrame(emb_month_size, 
                                               emb_weekday_size, 
                                               num_features, 
                                               lstm_hidden, 
                                               num_layers, 
                                               n_heads, 
                                               dropout)
        
        # Блок предсказания
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len)
        )

    def forward(self, x, time):
        """
        x: [batch_size, seq_len, num_features-5] - основные фичи
        time: [batch_size, seq_len, 5] - [месяц, день, час, минута, день_недели]
        """
        context_vector = self.LSTMTimeFrameLare(x, time)

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