from typing import List, Generator
import copy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset

from backend.Dataset import LoaderTimeLine
from backend.MMM import Agent

class BatchGenerator:
    def __init__(self, loaders: List[LoaderTimeLine], batch_size: int, mixed: bool = False):
        self.loaders = loaders
        self.batch_size = batch_size
        self.mixed = mixed
        self.peeked = None

        self.generator = self.batch_generator(loaders, batch_size, mixed)

    def peek(self):
        if self.peeked is None:
            try:
                self.peeked = next(self.generator)
            except StopIteration:
                self.peeked = None

        return self.peeked

    def __iter__(self):
        return self

    def __next__(self):
        if self.peeked is not None:
            value = self.peeked
            self.peeked = None
            return value
        
        return next(self.generator)

    @staticmethod
    def batch_generator(loaders: List[LoaderTimeLine], bath_size: int = 10, mixed: bool = False) -> Generator[None, None, List]:

        time_line_buffer = []

        loaders = [loader.get_loader() for loader in loaders]

        while loaders:
            for i, loader in enumerate(loaders):
                for time_line in loader:
                    time_line_buffer.append(time_line)
                    if mixed:
                        break
                    else:
                        if len(time_line_buffer) == bath_size:
                            yield time_line_buffer
                            time_line_buffer = []

                else:
                    loaders.pop(i)

                if not mixed or not time_line_buffer:
                    loaders.pop(i)
                    break
        
                if len(time_line_buffer) == bath_size:
                    yield time_line_buffer
                    time_line_buffer = []

        if time_line_buffer and len(time_line_buffer) == bath_size:
            yield time_line_buffer

class TimeSeriesTransform(IterableDataset):

    def __init__(self, loaders: List[LoaderTimeLine], agent: Agent, 
                 batch_size: int, 
                 mixed: bool = False):
        
        self.loaders = loaders
        self.agent = agent
        self.seq_len = agent.model_parameters["seq_len"]
        self.pred_len = agent.model_parameters["pred_len"]
        self.batch_size = batch_size
        self.mixed = mixed

        self._gen = BatchGenerator(loaders=copy.deepcopy(self.loaders), 
                                          batch_size=self.batch_size, 
                                          mixed=self.mixed)
        self._len = None
    
    def __iter__(self) -> iter:
        if self._gen.peek() is None:
            self._gen = BatchGenerator(loaders=copy.deepcopy(self.loaders),
                                            batch_size=self.batch_size, 
                                            mixed=self.mixed)
        while self._gen.peek() is not None:
            # Получаем батч из генератора
            time_line = next(self._gen)
            time_line_buffer = []
            for data in time_line:

                data, time_features = self.agent.preprocess_data(data)
                n_samples = data.shape[0]

                data = data.values
                time_features = time_features.values
                
                for i in range(n_samples - self.pred_len - self.seq_len):
                    x = data[i:i+self.seq_len]
                    y = data[i+self.seq_len: i + self.seq_len + self.pred_len]
                    time_x = time_features[i:i+self.seq_len]
                    time_line_buffer.append((x, y, time_x))
                    
                    if len(time_line_buffer) == self.batch_size:
                        yield self._process_batch(time_line_buffer)
                        time_line_buffer = []
            
    def __len__(self):
        if self._len is None:
            self._len = len([data for data in self])

        return self._len
    
    def _process_batch(self, batch):
        # Векторизованная обработка батча
        x, y, time_x = zip(*batch)
        return (
            torch.as_tensor(np.stack(x), dtype=torch.float32),
            torch.as_tensor(np.stack(y), dtype=torch.float32),
            torch.as_tensor(np.stack(time_x), dtype=torch.float32)
        )