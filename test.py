# %%
# !pip install --upgrade pip
# !pip install matplotlib
# !pip install -U scikit-learn
# !pip install sktime
# !pip install torch torchvision torchaudio
# !pip install transformers
# !pip install pytesseract
# !pip install opencv-python

# %%
from core import data_manager, DataManager, settings
from core.utils import setup_logging
from pathlib import Path

# settings.logging.level = "DEBUG"

setup_logging()

# %%
# data_manager.backup_data(paths=[Path("/Users/igor/Desktop/App-parser/data/raw/")])

# %%
coins = {}

for coin in data_manager.coin_list:
    if coin != "TON":
        continue
    for path in data_manager.get_path(data_type="raw",
                                      coin=coin,
                                      timetravel="5m"):
        coins.setdefault(coin, [])
        coins[coin].append(path)

# %%
from backend.Dataset import DatasetTimeseries
import pandas as pd

# %%
coins

# %%
coins_dt = {}
coins_dubl = {}
for coin, paths in coins.items():
    for path in paths:
        dt = DatasetTimeseries(path)
        dt.clear_dataset()
        coins_dt.setdefault(coin, [])
        coins_dt[coin].append(dt)

    df_new = pd.concat(map(lambda dt: dt.get_dataset(), coins_dt[coin]), ignore_index=True)
    df_new = DatasetTimeseries(df_new).dataset_clear()
    df_new.drop_duplicates(subset=["datetime", "open", "min", "max", "close", "volume"], inplace=True)
    df_new = DatasetTimeseries(df_new)
    coins_dubl.setdefault(coin, len(df_new.duplicated()))

    df_new.set_dataset(df_new.clear_dataset())
    path_coin = data_manager.create_dir("processed", coin)
    df_new.set_path_save(data_manager.create_dir("processed", path_coin / "5m"))
    df_new.save_dataset(f"clear-{coin}-5m.csv")
    coins_dt[coin] = df_new
    

# %%
for coin, count in coins_dubl.items():
    if count == 0:
        continue
    print(f"Coin: {coin}, Duplicates: {count}")

# %%
for coin, dt in coins_dt.items():
    if (len(dt.get_dataset_Nan()) / len(dt.get_dataset())) < 0.3:
        print(f"{coin=}\n{dt.get_dataset().shape=}\n{dt.get_dataset_Nan().shape=}")
        print(len(dt.get_dataset_Nan()) / len(dt.get_dataset()))
        print()
    

# %%
# dt_TON = coins_dt["AVAX"]
# df: pd.DataFrame = dt_TON.get_dataset()
# # dubl = df.duplicated(subset=["datetime"], keep=False)
# for data in dt_TON:
#     if data["volume"] != "x" and data["volume"] > 100:
#         print(data)
    

# %%
from backend.MMM import Agent, AgentPReadTime
from backend.Dataset import Indicators
from backend.train_models import Loader


# %%
coin_loader = {}
for coin, dt in coins_dt.items():
    dt: DatasetTimeseries
    filter_func = lambda x: x["open"] != "x"

    loader = dt.get_time_line_loader(time_line_size=40, 
                                filter_data=filter_func)
    print(f"{coin=}, {len(loader)=}")
    coin_loader[coin] = loader

loader_train = Loader("agent_pred_train")
agent_manager = loader_train.load_model(count_agents=1000)


# %%
loader = loader_train.train_model(list(coin_loader.values()),
                                  agent_manager=agent_manager,
                                  mixed=True,)

# %%
print(type(loader))
count = 0
# loader_t = loader.get_loader()
flag = False
for bath in loader:
    x, y, time_x = bath
    count += 1
    if not flag:
        flag = True
        data = bath
        print(x.shape, y.shape, time_x.shape)
        break
        # print(x[0])
        # print(y[0])
        # print(time_x[0])
    # if time_x[0][0][0] < 2020:
    #     print(bath)
    # print(time_x)

print(count)
print(data)