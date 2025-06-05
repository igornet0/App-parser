import torch

from backend.MMM.models.model_trade import TradingModel

if __name__ == "__main__":
    # Генерация реалистичных временных данных
    time_data = torch.stack([
        torch.randint(1, 13, (32, 30)),         # Month
        torch.randint(1, 32, (32, 30)),         # Day
        torch.randint(0, 24, (32, 30)),         # Hour
        torch.randint(0, 60, (32, 30)),         # Minute
        torch.randint(0, 8, (32, 30)),          # Weekday
    ], dim=-1).float()
    
    model = TradingModel(30, 5, 5)
    pred_data = torch.randn(32, 30, 5)
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
    
    
    output = model(main_data, pred_data, time_data)


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