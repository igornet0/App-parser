from typing import Optional, Callable, Iterable
import pandas as pd

from core.utils.tesseract_img_text import timetravel_seconds_int

class LoaderTimeLine:

    def __init__(self, 
                 dataset: Iterable,
                 time_line_size,
                 filter_data: Optional[Callable] = lambda x: True,
                 transform_data: Optional[Callable] = lambda x: x):
        
        self.dataset = dataset
        self.time_line_size = time_line_size
        self.filter_data = filter_data
        self.transform_data = transform_data
        self.count = None

    def get_loader(self):
        time_line = []
        timedelta_seconds = timetravel_seconds_int[self.dataset.timetravel]

        for data in self.dataset:

            if not self.filter_data(data):
                continue

            if time_line and abs((time_line[-1]["datetime"] - data["datetime"]).total_seconds()) != timedelta_seconds:
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