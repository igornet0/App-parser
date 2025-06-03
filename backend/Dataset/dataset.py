from __future__ import annotations

from sktime import utils

import pandas as pd
from typing import Generator, List, Dict, Callable, Optional
from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import PosixPath
from urllib.parse import urlparse
from typing import Union
from os import walk, mkdir, path, getcwd
import re

from torch.utils.data import Dataset as _Dataset, DataLoader
from transformers import BertTokenizer
import torch

from core import data_manager
from core.utils.clear_datasets import *
from core.utils.tesseract_img_text import RU_EN_timetravel
from .loader import LoaderTimeLine

import logging

logger = logging.getLogger("Dataset")

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time}")
        return result
    
    return wrapper

class Dataset(_Dataset):

    def __init__(self, dataset: Union[pd.DataFrame, dict, str], transforms=None, target_column: str=None) -> None:
        
        if isinstance(dataset, str) or isinstance(dataset, PosixPath):
            path_open = self.searh_path_dateset(dataset)

            if isinstance(path_open, list):
                raise FileNotFoundError(f"File {dataset} not found in {getcwd()}")
    
            dataset = pd.read_csv(path_open)
            self.set_filename(str(path_open).split("/")[-1])

        elif not isinstance(dataset, pd.DataFrame):
            logger.error(f"Invalid dataset type {type(dataset)}")
            self.set_filename("clear_dataset.csv")
        
        self.drop_unnamed(dataset)

        if "date" in dataset.columns:
            dataset.rename(columns={"date": "datetime"}, inplace=True)

        if "datetime" in dataset.columns:
            dataset["datetime"] = pd.to_datetime(dataset["datetime"], format='%Y-%m-%d %H:%M:%S')

        self.dataset = dataset
        self.transforms = transforms

        if target_column:
            self.targets = dataset[target_column]
            self.dataset.drop(target_column, axis=1, inplace=True)
        else:
            self.targets = None

        self.path_save = data_manager["processed"]

    def get_datetime_last(self) -> datetime:
        return self.dataset['datetime'].iloc[-1]
    
    def set_path_save(self, path_save: str) -> None:
        self.path_save = path_save

    def set_filename(self, file_name: str) -> None:
        self.file_name = file_name

    def get_filename(self) -> str:
        return self.file_name

    def get_dataset(self) -> pd.DataFrame:
        return self.dataset
    
    def set_dataset(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
    
    def get_data(self, idx: int):
        return self.dataset.iloc[idx]
    
    @timer
    def clear_dataset(self) -> pd.DataFrame:
        return clear_dataset(self.dataset)

    @classmethod
    def drop_unnamed(cls, dataset):
        try:
            dataset.drop('Unnamed: 0', axis=1, inplace=True)
        except Exception:
            pass

    @classmethod
    def searh_path_dateset(cls, pattern: str, root_dir=getcwd()) -> list[str]:
        # Преобразуем шаблон в регулярное выражение
        if path.exists(pattern) and path.isfile(pattern):
            return pattern

        regex_pattern = '^' + '.*'.join(re.escape(part) for part in pattern.split('*')) + '$'
        regex = re.compile(regex_pattern)
        
        matched_files = []
        
        for dirpath, _, filenames in walk(root_dir):
            for filename in filenames:
                if regex.match(filename):
                    full_path = path.join(dirpath, filename)
                    matched_files.append(full_path)

        if not matched_files:
            raise FileNotFoundError(f"File {pattern} not found in {root_dir}")
        
        return matched_files
    
    @classmethod
    def concat_dataset(cls, *dataset: pd.DataFrame | Dataset) -> pd.DataFrame:
        return pd.concat([data.get_dataset() if isinstance(data, Dataset) else data for data in dataset], ignore_index=True)
    
    def save_dataset(self, name_file: str = None) -> None:
        if not path.exists(self.path_save):
            mkdir(self.path_save)

        if name_file is None:
            name_file = self.file_name

        # if path.exists(path.join(self.path_save, name_file)):
        #     dataset = Dataset(path.join(self.path_save, name_file))
        #     self.concat_dataset(self.dataset, dataset)

        self.dataset.to_csv(path.join(self.path_save, name_file), index=False, encoding='utf-8')
        logger.info(f"Dataset saved to {path.join(self.path_save, name_file)}")

    def __iter__(self):
        for index, data in self.dataset.iterrows():
            yield data

    def __getitem__(self, idx: int):
            
        sample = self.get_data(idx)
        
        if self.transforms:
            sample = self.transforms(sample)

        if self.targets:
            target = self.targets.iloc[idx]
            target = torch.tensor(target, dtype=torch.long)  
            return sample, target

        return sample, self.targets

    def __len__(self):
        return len(self.dataset)


class DatasetTimeseries(Dataset):
    
    def __init__(self, dataset: Union[pd.DataFrame, dict, str] , timetravel: str = "5m") -> None:
        
        super().__init__(dataset)

        if "datetime" not in self.dataset.columns and "date" in self.dataset.columns:
            self.dataset.rename(columns={"date": "datetime"}, inplace=True)

        elif "datetime" not in self.dataset.columns and "date" not in self.dataset.columns:
            raise ValueError("Columns 'datetime' or 'date' not found in dataset")
        elif "open" not in self.dataset.columns:
            raise ValueError("Column 'open' not found in dataset")
        elif "close" not in self.dataset.columns:
            raise ValueError("Column 'close' not found in dataset")
        elif "max" not in self.dataset.columns:
            raise ValueError("Column 'max' not found in dataset")
        elif "min" not in self.dataset.columns:
            raise ValueError("Column 'min' not found in dataset")
        elif "volume" not in self.dataset.columns:
            raise ValueError("Column 'volume' not found in dataset")
        
        # self.dataset["datetime"] = self.dataset["datetime"].apply(safe_convert_datetime)
        self.dataset["datetime"] = pd.to_datetime(self.dataset["datetime"], 
                                                  format='%Y-%m-%d %H:%M:%S', 
                                                  errors='coerce')
        
        self.dataset["open"] = self.dataset["open"].apply(str_to_float)
        self.dataset["close"] = self.dataset["close"].apply(str_to_float)
        self.dataset["max"] = self.dataset["max"].apply(str_to_float)
        self.dataset["min"] = self.dataset["min"].apply(str_to_float)
        self.dataset["volume"] = self.dataset["volume"].apply(str_to_float)
    
        self.dataset = self.dataset.dropna(subset=["datetime"])

        self.timetravel = timetravel

    def get_time_line_loader(self, time_line_size: int = 30, 
                        filter_data: Optional[Callable] = lambda x: True,
                        transform_data: Optional[Callable] = lambda x: x) -> LoaderTimeLine:
        
        self.dataset = self.dataset.sort_values(by="datetime", 
                                        ignore_index=True)
        
        return LoaderTimeLine(self, time_line_size, filter_data, transform_data, self.timetravel)

    def get_count_time_line(self, time_line_size: int = 30, 
                        filter_data: Optional[Callable] = lambda x: True,
                        transform_data: Optional[Callable] = lambda x: x) -> int:
        count = 0
        for _ in self.get_time_line_loader(time_line_size, filter_data, transform_data):
            count += 1
        return count
            
    @timer
    def sort(self, column: str = "datetime"):
        self.dataset = self.dataset.sort_values(by=column, 
                                        ignore_index=True,
                                        ascending=True)
        return self

    @timer
    def clear_dataset(self) -> pd.DataFrame:
        # self.dataset = clear_dataset(self.dataset, sort=True, timetravel=self.timetravel)
        dataset = self.dataset.copy()

        for col in dataset.columns:
            if col in ["datetime", "volume"]:
                continue

            dataset[col] = dataset[col].apply(str_to_float) 

        dataset = convert_volume(dataset)
        logger.debug("Volume converted to float %d", len(dataset))

        # dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)
        grouped = find_most_common_df(dataset)

        dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)
        dataset = pd.concat([grouped, dataset])

        dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)

        dataset = conncat_missing_rows(dataset, timetravel=self.timetravel)
        
        logger.debug("Missing rows concatenated %d", len(dataset))

        dataset = dataset.drop_duplicates(subset=['datetime'], ignore_index=True)
        logger.debug("Duplicates removed %d", len(dataset))

        dataset = dataset.sort_values(by='datetime', 
                                        ignore_index=True,
                                        ascending=False)
        logger.debug("Dataset sorted %d", len(dataset))
        # self.dataset = dataset
        
        return dataset
    
    def set_timetravel(self, timetravel: str):
        if not (timetravel in RU_EN_timetravel.keys() or timetravel in RU_EN_timetravel.values()):
            raise ValueError(f"Invalid timetravel: {timetravel}")

        self.timetravel = timetravel
    
    def duplicated(self):
        return self.dataset[self.dataset.duplicated(keep=False)]

    def plot_series(self, dataset: list | None = None, param: str = "close", indicators: dict | None = None) -> None:
        # Determine main dataset, dates, and data
        if dataset is None:
            main_data = self.dataset[param]
            if 'datetime' in self.dataset.columns:
                main_dates = self.dataset['datetime']
            else:
                main_dates = main_data.index
        else:
            main_dates = dataset['datetime']
            main_data = dataset[param]

        # Check if there are indicators to plot
        if indicators is None:
            # Original plotting without indicators
            if dataset is None:
                plt.figure(figsize=(12, 8))
                plt.plot(main_dates, main_data)
                plt.title(param)
                plt.tick_params(axis='both', which='major', labelsize=14)
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(10, 5))
                plt.plot(main_dates, main_data, marker='o')
                plt.xlabel('Время')
                plt.ylabel('Цена')
                plt.xticks(rotation=45)
                plt.grid()
                plt.tight_layout()
                plt.show()
            return

        # Split indicators into those on the main plot and separate subplots
        on_indicators = {}
        off_indicators = {}
        for name, ind in indicators.items():
            if ind.get('on', False):
                on_indicators[name] = ind
            else:
                off_indicators[name] = ind

        num_subplots = 1 + len(off_indicators)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots), sharex=True)
        if num_subplots == 1:
            axes = [axes]  # Ensure axes is a list for consistent handling

        # Plot main data and on=True indicators on the first subplot
        ax_main = axes[0]
        ax_main.plot(main_dates, main_data, label=param, marker='o' if dataset is not None else None)
        for name, ind in on_indicators.items():
            ax_main.plot(main_dates, ind['data'], label=name)

        ax_main.set_title(param)
        ax_main.grid(True)
        ax_main.legend()
        ax_main.tick_params(axis='both', labelsize=12)

        # Plot off_indicators on subsequent subplots
        for i, (name, ind) in enumerate(off_indicators.items(), start=1):
            ax = axes[i]
            ax.plot(main_dates, ind['data'], label=name)
            ax.grid(True)
            ax.legend()
            ax.set_ylabel(name)
            ax.tick_params(axis='both', labelsize=12)

        # Set common x-axis labels
        axes[-1].set_xlabel('Время' if dataset is not None else 'Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_dataset_Nan(self) -> pd.DataFrame:
        return self.dataset.loc[self.dataset['open'] == "x"]
    
    def dataset_clear(self) -> pd.DataFrame:
        return self.dataset.loc[self.dataset['open'] != "x"]
    
    def get_datetime_last(self) -> datetime:
        return self.dataset['datetime'].iloc[-1]


class NewsDataset(Dataset):

    def __init__(self, dataset, file_path: str, targets=None, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), max_len=128):

        """
        Parameters
        ----------
        file_path : str
            Path to file with news texts
        targets : list or None, default=None
            List of targets (labels or values for regression)
        tokenizer : PreTrainedTokenizer, default=BertTokenizer.from_pretrained('bert-base-uncased')
            BERT tokenizer
        max_len : int, default=128
            Maximum length of tokens
        """

        if path.exists(file_path):
            file_name = DatasetTimeseries.searh_dateset(dataset)
            file_path = path.join(dataset, file_name)
            self.news = pd.read_csv(path.join(dataset, file_name),
                                    parse_dates=["datetime"])
            
            if 'Unnamed: 0' in dataset.columns:
                self.news.drop('Unnamed: 0', axis=1, inplace=True)
        else:
            raise ValueError("File not found")

        self.file_path = file_path          
        self.targets = targets                # Целевые значения (метки или значения для регрессии)
        self.tokenizer = tokenizer            # Токенизатор BERT
        self.max_len = max_len                # Максимальная длина токенов
    

    def get_loader(self):
        return DataLoader(self, batch_size=2, shuffle=True)

    
    @classmethod
    def get_domains(cls, news: pd.DataFrame):
        if "url" in news.columns:
            column = "url"
        elif "news_url" in news.columns:
            column = "news_url"
        else:
            return None
        return news[column].apply(lambda x: urlparse(x).netloc).unique().tolist()

    def get_news(self):
        return self.news

    def __len__(self):
        return len(self.news)
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self.news.iloc[idx]
    
    def __getitem__(self, idx):
        # Получаем текст новости и его цену по индексу
        news_text = self.news["text"].iloc[idx]
        # prices = self.price_data[idx]
        # target = self.targets[idx]
        
        # Токенизация текста с помощью BERT токенизатора
        encoding = self.tokenizer.encode_plus(
            news_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Возвращаем закодированные данные текста, цен и целевую метку
        return {
            'news_input_ids': encoding['input_ids'].flatten(),  # Извлекаем из тензора
            'news_attention_mask': encoding['attention_mask'].flatten(),
            # 'price_data': prices.float(),  # Преобразуем в float для LSTM
            # 'target': torch.tensor(target, dtype=torch.float)  # Целевое значение (например, для регрессии)
        }