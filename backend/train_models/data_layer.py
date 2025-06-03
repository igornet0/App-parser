import pandas as pd

def sear_datetime(dt: pd.DataFrame, date):
    return dt[dt["datetime"] == date]

class ProccesedData:

    def __init__(self, data_correct: dict[str, list[pd.DataFrame]]):
        self.data_correct = data_correct

    def get_min_max_date(self, dts):
        min_data = None
        max_data = None
        for dt in dts:
            if min_data is None or min_data > dt.get_dataset().datetime.min():
                min_data = dt.get_dataset().datetime.min()
            if max_data is None or max_data < dt.get_dataset().datetime.max():
                max_data = dt.get_dataset().datetime.max()

        return min_data, max_data

    def procces(self, data_procces: dict[str, list[pd.DataFrame]]):
        coin_new_dt = {}

        for coin, dts in data_procces.items():
            if coin in ["SHIB"]:
                continue
    
            print("-"* 10)
            print(coin)

            min_data, max_data = self.get_min_max_date(dts)
            
            if (min_data - max_data).total_seconds() == 0:
                continue

            print(f"min: {min_data}")
            print(f"max: {max_data}")
            print((max_data - min_data).total_seconds())

            new_dt = []
            correct_dt = {}
            
            for date in pd.date_range(min_data, max_data, freq="5min"):
                data_buffer = {}

                for dt in dts:
                    result = sear_datetime(dt.get_dataset(), date)

                    if len(result) == 0:
                        continue

                    data_buffer.setdefault(dt, result)

                if len(data_buffer.values()) > 1:
                    datas = list(data_buffer.values())
                    if len(datas) == 2:
                        if datas[0]["open"].item() != "x" and datas[1]["open"].item() == "x":
                            new_dt.append(datas[0])
                        elif datas[0]["open"].item() == "x" and datas[1]["open"].item() != "x":
                            new_dt.append(datas[1])
                        continue

                    for data in datas[1:]:
                        if data["volume"].item() == datas[0]["volume"].item():
                            continue
                        i = 0
                        dt_i = {}
                        i_dt = {}
                        flag_correct = False
                        dt_correct = None
                        for dt, result in data_buffer.items():
                            print(f"{i}: {result}")
                            dt_i[i] = dt
                            i_dt[dt] = i
                            i += 1

                        for dt, result in data_buffer.items():
                            if correct_dt.get(dt, 0) > 50:
                                print("-"* 10)
                                print(f"Coorect - {i_dt[dt]}{dt}: {correct_dt[dt]}")
                                print("-"* 10)
                                flag_correct = True
                                dt_correct = dt
                                break

                        s = input()

                        if s == "c":
                            break
                        
                        elif flag_correct and s == "y" and dt_correct is not None:
                            correct_dt.setdefault(dt, 0)
                            correct_dt[dt] += 1
                            new_dt.append(data_buffer[dt_correct])
                            break
                        elif s.isdigit():
                            dt = dt_i[int(s)]
                            correct_dt.setdefault(dt, 0)
                            correct_dt[dt] += 1
                            new_dt.append(datas[int(s)])
                            break
                        print("-"* 10)

                else:
                    new_dt.append(list(data_buffer.values())[0])

            coin_new_dt[coin] = new_dt

        return coin_new_dt