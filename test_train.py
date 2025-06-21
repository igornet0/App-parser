# %%
from core import data_helper, settings
from core.utils import setup_logging
from backend.Dataset import DatasetTimeseries

settings.logging.level = "DEBUG"

setup_logging()

coins = {}

for coin in data_helper.coin_list:
    for path in data_helper.get_path(data_type="processed",
                                      coin=coin,
                                      timetravel="5m"):
        coins.setdefault(coin, [])
        coins[coin].append(path)

# raw_coins = {}

# for coin in data_manager.coin_list:
#     for path in data_manager.get_path(data_type="raw",
#                                       coin=coin,
#                                       timetravel="5m"):
#         raw_coins.setdefault(coin, [])
#         raw_coins[coin].append(path)

# %%
coins_dt = {}
for i, data in enumerate(coins.items()):
    coin, paths = data
    for path in paths:
        dt = DatasetTimeseries(path)
        coins_dt.setdefault(coin, [])
        coins_dt[coin].append(dt)

# %%
for coin, dts in coins_dt.items():

    if len(dts) == 1:
        coins_dt[coin] = dts[0]
        continue
    else:
        print(coin, len(dts))
        continue
    df = DatasetTimeseries.concat_dataset(*dts)
    df.drop_duplicates(subset=["datetime", "open", "min", "max", "close", "volume"], inplace=True)
    df.drop_duplicates(subset=["datetime"], inplace=True)
    df = DatasetTimeseries(df).dataset_clear()
    df = DatasetTimeseries(df)

    coins_dt[coin] = df

# %%
for coin, dts in coins_dt.items():
    df = coins_dt[coin].get_dataset()
    dubl = df[df.duplicated(subset=["datetime"])]
    if len(dubl) == 0:
        continue
    print(coin)
    print(len(dubl))


# %%
coins_loader = {}

times = {"5m": 300, "15m": 250, "30m": 200, "1H": 150, "4H": 20, "1D": 7}

def filter_func(x):
    if x["open"] != "x" and isinstance(x["open"], str):
        print("ERROR !!!!!! - ", x)
        return True
    return x["open"] != "x"

for coin, dt_clear in coins_dt.items():
    for time, size in times.items():
        dt_clear: DatasetTimeseries
        dt_clear.set_timetravel(time)
        # filter_func = lambda x: x["open"] != "x"

        loader_time_line = dt_clear.get_time_line_loader(time_line_size=size, filter_data=filter_func)

        coins_loader.setdefault(coin, {})
        coins_loader[coin][time] = loader_time_line

# %%
#LAst data 
#{'5m': 16213, '15m': 10268, '30m': 10254, '1H': 0, '4H': 0, '1D': 0}

# count_time_line = {}
# for coin, time_dt in coins_loader.items():
#     # print(coin)
#     for time, loader in time_dt.items():
#         count_time_line.setdefault(time, 0)
#         count_time_line[time] += len(loader)
        # print(time, len(loader))
    # print("-"* 10)

# print(count_time_line)

# %%
from backend.train_models import Loader
from backend.MMM import AgentManager, Agent
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from accelerate import Accelerator

model_name = "agent_pred_train_2"
agent_type = "AgentPredTime"

loader_train = Loader(agent_type, model_name)
agent_manager = loader_train.load_model(count_agents=30)
# %%
config_train = data_helper.get_model_config(agent_type, model_name)
        
# Конфигурация обучения
epochs = config_train["epochs"]
batch_size = config_train["batch_size"]
num_workers = config_train.get("num_workers", 4)
base_lr = config_train["lr"]
weight_decay = config_train["weight_decay"]
patience = config_train["patience"]
mixed_precision = config_train.get("mixed_precision", False)

print("epochs:", epochs)
print("batch_size:", batch_size)

# %%
# loader_time = []

# for coin, time_dt in coins_loader.items():
#     # if coin not in ["BTC", "TON", "SOL", "XRP", "BNB", "XMR", "ETH"]:
#     if coin not in ["BTC", "ETH"]:
#         continue
#     loader_time.append(time_dt["5m"])

#     data, time_features = agent.preprocess_data_for_model(data, normalize=True)
#     n_samples = data.shape[0]

#     data = data.values
#     time_features = time_features.values
    
#     for i in range(n_samples - pred_len - seq_len):
#         x = data[i:i+seq_len]
#         y = data[i+seq_len: i + seq_len + pred_len]
#         y = y.reshape(-1, 1)  # Приводим к нужной форме

# %%
loader_time = []
for coin, time_dt in coins_loader.items():
    # if coin not in ["BTC", "TON", "SOL", "XRP", "BNB", "XMR", "ETH"]:
    #     continue
    # if coin not in ["BTC", "TON"]:
    #     continue
    loader_time.append(time_dt["5m"])

# %%
mixed = True
pred_len = 5
seq_len = 30
agents_all = agent_manager.get_agents()
for time, agents in agents_all.items():
    if time != "5m":
        continue

    print(time, len(agents))
    for i, agent in enumerate(agents):
        print(agent)
        # loader = loader_train.load_agent_data(loader_time, agent, batch_size, mixed)
        # for x,y,time in loader:
        #     # y = y.squeeze(dim=-1) 
        #     print(x, time)
        #     print(y)
        #     break
        # break
        loader_train._train_single_agent(agent, loader_time, epochs, batch_size, 
                                 base_lr, weight_decay, patience, mixed, 
                                 mixed_precision)
    break

#8175

# %%
# mixed = True
# pred_len = 5
# seq_len = 30
# agents_all = agent_manager.get_agents()
# time_buffer = []
# for time, agents in agents_all.items():
#     if time != "5m":
#         continue

#     print(time, len(agents))
#     for i, agent in enumerate(agents):
#         print(agent)
#         print("count_input_features:", agent.get_count_input_features())
#         print("shape_indecaters:", agent.get_shape_indecaters())
#         loader = loader_train.load_agent_data(loader_time, agent, batch_size, mixed)
#         # count = 0
#         for x, y, time in loader:
#             print("X = ", x.shape," Time =", time.shape)
#             # print(x[0])
#             print("Y = ",y.shape)
#             break
#         # print(count)
#         break
#     break


