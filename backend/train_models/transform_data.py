from typing import List, Generator, Tuple
import copy
import pandas as pd
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

    def batch_generator(self, loaders: List[LoaderTimeLine], bath_size: int = 10, mixed: bool = False) -> Generator[None, None, List]:

        time_line_buffer = []

        loaders = [loader.get_loader() for loader in loaders]
        
        while loaders:

            for i, loader in enumerate(loaders):
                loader_end = len(time_line_buffer)
                for time_line in loader:
                    
                    time_line_buffer.append(time_line)

                    if mixed:
                        break
                    else:
                        if len(time_line_buffer) == bath_size:
                            yield time_line_buffer
                            time_line_buffer = []

                if len(time_line_buffer) == bath_size:
                    yield time_line_buffer
                    time_line_buffer = []
                    continue
                
                if loader_end == len(time_line_buffer) or not time_line_buffer:
                    loaders.pop(i)
                    break

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
        
        self.time_line_loaders = {}
        self._gen = self.create_gen_batch(loaders, batch_size, mixed)
        self._len = None

    def get_loaders(self):
        return self.loaders

    def create_gen_batch(self, loaders: List, batch_size, mixed) -> BatchGenerator:
        return BatchGenerator(loaders=copy.deepcopy(loaders), 
                                batch_size=batch_size, 
                                mixed=mixed)
    
    def load_time_line(self, time_line):
        for i, data in enumerate(time_line):
            self.time_line_loaders[i] = self.agent.create_time_line_loader(data, self.pred_len, self.seq_len)

    def __iter__(self) -> iter:
        if self._gen.peek() is None:
            self._gen = self.create_gen_batch(self.loaders, self.batch_size, self.mixed)
        
        bath_data = []
        
        while self._gen.peek() is not None:
            
            time_line = next(self._gen) 
            
            if not len(self.time_line_loaders):
                self.load_time_line(time_line)
            
            while len(self.time_line_loaders):

                for i, batch in self.time_line_loaders.items():
                    
                    len_bath = len(bath_data)

                    for data in batch:

                        bath_data.append(data)
                        
                        if self.mixed:
                            break

                        if len(bath_data) == self.batch_size:
                            
                            yield self.agent.process_batch(bath_data)
                            bath_data = []


                    if len(bath_data) == self.batch_size:
                        yield self.agent.process_batch(bath_data)
                        bath_data = []
                        continue
                    
                    if len(bath_data) - len_bath == 0 or not bath_data:
                        self.time_line_loaders.pop(i)
                        break

        
    def __len__(self):
        if self._len is None:
            self._len = len([1 for _ in self])

        return self._len