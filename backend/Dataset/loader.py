from typing import Optional, Callable, Iterable, Generator
from collections import namedtuple
import pandas as pd

from core.utils.tesseract_img_text import timetravel_seconds_int

Data = namedtuple('DataTimeseries' , 'datetime open max min close volume')

class LoaderTimeLine:

    def __init__(self, 
                 dataset: Iterable,
                 time_line_size,
                 filter_data: Optional[Callable] = lambda x: True,
                 transform_data: Optional[Callable] = lambda x: x,
                 timetravel: str = "5m"):
        
        self.dataset = dataset
        self.time_line_size = time_line_size
        self.filter_data = filter_data
        self.transform_data = transform_data
        self.timetravel = timetravel
        self.count = None

    @staticmethod
    def check_timetravel(data: pd.DataFrame, timetravel: str = "5m") -> bool:
        if "m" in timetravel:
            timetravel = int(timetravel.replace("m", ""))
            if data["datetime"].minute % timetravel == 0:
                return True
        elif "h" in timetravel:
            timetravel = int(timetravel.replace("h", ""))
            if data["datetime"].hour % timetravel == 0 and data["datetime"].minute == 0:
                return True
        elif "d" in timetravel:
            timetravel = int(timetravel.replace("d", ""))
            if data["datetime"].day % timetravel == 0 and data["datetime"].hour == 0 and data["datetime"].minute == 0:
                return True
        elif "M" in timetravel:
            timetravel = int(timetravel.replace("M", ""))
            if data["datetime"].month % timetravel == 0 and data["datetime"].day == 1 and data["datetime"].hour == 0 and data["datetime"].minute == 0:
                return True
        elif "y" in timetravel:
            timetravel = int(timetravel.replace("y", ""))
            if data["datetime"].year % timetravel == 0 and data["datetime"].month == 1 and data["datetime"].day == 1 and data["datetime"].hour == 0 and data["datetime"].minute == 0:
                return True
            
        return False

    def bath_timetravel(self, dataset: pd.DataFrame, timetravel: str = "5m") -> Generator[pd.Series, None, None]:
        data_t = []
        for data in dataset:
            if not self.filter_data(data):
                data_t = []
                continue

            if LoaderTimeLine.check_timetravel(data, timetravel):
                if data_t:
                    new_row = {}
                    new_row["datetime"] = data_t[0]["datetime"]
                    new_row["open"] = data_t[0]["open"]
                    new_row["max"] = max(data_t, key=lambda x: x["max"])["max"]
                    new_row["min"] = min(data_t, key=lambda x: x["min"])["min"]
                    new_row["close"] = data_t[-1]["close"]
                    new_row["volume"] = sum(map(lambda x: float(x["volume"]), data_t))
                    new_row = pd.Series(new_row)
                    yield new_row
                    data_t = []

                data_t.append(data)
            else:
                if data_t:
                    data_t.append(data)

    def get_data(self) -> list[Data]:

        if self.timetravel != "5m":
            self.dataset = self.bath_timetravel(self.dataset, self.timetravel)

        time_line = []
        for data in self.dataset:
            time_line.append(Data(data["datetime"], data["open"], data["max"], data["min"], data["close"], data["volume"]))

        return time_line

    def get_loader(self) -> Generator[pd.DataFrame, None, None]:
        time_line = []
        timedelta_seconds = timetravel_seconds_int[self.timetravel]

        if self.timetravel != "5m":
            self.dataset = self.bath_timetravel(self.dataset, self.timetravel)

        for data in self.dataset:

            if not self.filter_data(data):
                continue

            if time_line and abs((time_line[-1]["datetime"] - data["datetime"]).total_seconds()) != timedelta_seconds:
                if len(time_line) == self.time_line_size:
                    yield self.transform_data(pd.DataFrame(time_line))

                time_line = []
            
            time_line.append(data)
            
            if len(time_line) == self.time_line_size:
                yield self.transform_data(pd.DataFrame(time_line))
                time_line = []
        
        if len(time_line) == self.time_line_size:
            yield self.transform_data(pd.DataFrame(time_line))

    def __iter__(self):
        return self.get_loader()

    def __len__(self):
        if self.count is None:
            count = 0
            for _ in self:
                count += 1
            self.count = count

        return self.count