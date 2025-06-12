
# # %%
# from core import data_manager, data_manager, settings
# from core.utils import setup_logging
# from pathlib import Path

# settings.logging.level = "DEBUG"

# setup_logging()

# # %%
# # data_manager.backup_data(paths=[Path("/Users/igor/Desktop/App-parser/data/raw/")])

# # %%
# coins = {}

# for coin in data_manager.coin_list:
#     # if coin != "TON":
#     #     continue
#     for path in data_manager.get_path(data_type="raw",
#                                       coin=coin,
#                                       timetravel="5m"):
#         coins.setdefault(coin, [])
#         coins[coin].append(path)

# # %%
# from backend.Dataset import DatasetTimeseries
# import pandas as pd

# # %%
# coins

# # %%
# coins_dt = {}
# coins_dubl = {}
# for coin, paths in coins.items():
#     for path in paths:
#         dt = DatasetTimeseries(path)
#         dt.set_dataset(dt.clear_dataset())
#         coins_dt.setdefault(coin, [])
#         coins_dt[coin].append(dt)

#     # df_new = pd.concat(map(lambda dt: dt.get_dataset(), coins_dt[coin]), ignore_index=True)
#     # df_new = DatasetTimeseries(df_new).dataset_clear()
#     # df_new.drop_duplicates(subset=["datetime", "open", "min", "max", "close", "volume"], inplace=True)
#     # df_new = DatasetTimeseries(df_new)
#     # coins_dubl.setdefault(coin, len(df_new.duplicated()))

#     # df_new.set_dataset(df_new.clear_dataset())
#     # path_coin = data_manager.create_dir("processed", coin)
#     # df_new.set_path_save(data_manager.create_dir("processed", path_coin / "5m"))
#     # df_new.save_dataset(f"clear-{coin}-5m.csv")
#     # coins_dt[coin] = df_new
    

# # %%
# def sear_datetime(dt: pd.DataFrame, date):
#     return dt[dt["datetime"] == date]


# # %%
# coin_new_dt = {}

# for coin, dts in coins_dt.items():
#     min_data = None
#     max_data = None
#     print("-"* 10)
#     print(coin)
#     for dt in dts:
#         if min_data is None or min_data > dt.get_dataset().datetime.min():
#             min_data = dt.get_dataset().datetime.min()
#         if max_data is None or max_data < dt.get_dataset().datetime.max():
#             max_data = dt.get_dataset().datetime.max()
#     if (min_data - max_data).total_seconds() == 0:
#         continue
#     print(f"min: {min_data}")
#     print(f"max: {max_data}")
#     print((min_data - max_data).total_seconds())

#     new_dt = []
#     correct_dt = {}
    
#     for date in pd.date_range(min_data, max_data, freq="5min"):
#         data_buffer = {}
#         for dt in dts:
#             result = sear_datetime(dt.get_dataset(), date)
#             if len(result) == 0:
#                 continue
#             data_buffer.setdefault(dt, result)

#         if len(data_buffer.values()) > 1:
#             datas = list(data_buffer.values())
#             for data in datas[1:]:
#                 if data["volume"].item() == datas[0]["volume"].item():
#                     continue
#                 i = 0
#                 dt_i = {}
#                 i_dt = {}
#                 flag_correct = False
#                 dt_correct = None
#                 for dt, result in data_buffer.items():
#                     print(f"{i}: {result}")
#                     dt_i[i] = dt
#                     i_dt[dt] = i
#                     i += 1

#                 for dt, result in data_buffer.items():
#                     if correct_dt[dt] > 50:
#                         print("-"* 10)
#                         print(f"Coorect - {i_dt[dt]}{dt}: {correct_dt[dt]}")
#                         print("-"* 10)
#                         flag_correct = True
#                         dt_correct = dt
#                         break

#                 s = input()

#                 if s == "c":
#                     break
                
#                 elif flag_correct and s == "y" and dt_correct is not None:
#                     correct_dt.setdefault(dt, 0)
#                     correct_dt[dt] += 1
#                     new_dt.append(data_buffer[dt_correct])
#                     break
#                 elif s.isdigit():
#                     dt = dt_i[int(s)]
#                     correct_dt.setdefault(dt, 0)
#                     correct_dt[dt] += 1
#                     new_dt.append(datas[int(s)])
#                     break
#                 print("-"* 10)

#         else:
#             new_dt.append(list(data_buffer.values())[0])

#     coin_new_dt[coin] = new_dt
import asyncio

def f(n):
    if n <= 1:
        return 1
    return f(n - 1) + f(n - 2)


async def process(i):
    a = 0
    for n in range(i):
        a += f(n)
    return a

class Process:
    def __init__(self, i):
        self.i = i

    async def run(self):
        return await process(self.i)

class ManagerProcess:

    proccess = []
    result = []
    cpu = 3

    async def start_parser(self):
        buffer_process = []

        for process in self.process:
            if len(buffer_process) == self.cpu:
                await self._manager_process(buffer_process)
                buffer_process = []

            buffer_process.append(process)

        while buffer_process:
            try:
                next(self._manager_process(buffer_process))
            except StopIteration:
                break
    
    def _init_manag(self):
        pass

    async def _manager_process(self, buffer_process):
        for process in buffer_process:
            await process.run()

    def add_process(self, process):
        self.process.append(process)

    def run(self):
        for process in self.process:
            process.run()