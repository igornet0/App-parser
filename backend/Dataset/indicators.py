import pandas as pd
import numpy as np

class Indicators:

    collumns_shape = {
        'SMA': "SMA{period}",
        'EMA': "EMA{period}",
        'RSI': "RSI{period}",
        'MACD': ["MACD", "Signal"],
        'BOLLINGER': ["UpperBB", "MiddleBB", "LowerBB"],
        'ATR': "ATR{period}",
        'STOCHASTIC_OSCILLATOR': ["%K", "%D"],
        'VWAP': "VWAP",
        'OBV': "OBV",
        "MFI": "MFI{period}",
        "CRV": "CRV{period}{trading_days}"
    }

    collumns_on = {
        'SMA': True,
        'EMA': True,
        'VWAP': True
    }

    @classmethod
    def check_indecater_on(cls, name_indecater: str) -> bool:
        return cls.collumns_on.get(name_indecater, False)
    
    @staticmethod
    def _check_data(data: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['datetime', 'open', 'max', 'min', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("DataFrame must contain all required columns")
        
        return data.sort_values('datetime')
    
    @staticmethod
    def sma(data: pd.DataFrame, period=14, column='close'):
        """Простая скользящая средняя (Simple Moving Average)"""
        data = Indicators._check_data(data)
        return data[column].rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.DataFrame, period=14, column='close'):
        """Экспоненциальная скользящая средняя (Exponential Moving Average)"""
        data = Indicators._check_data(data)
        return data[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.DataFrame, period=14):
        """Индекс относительной силы (Relative Strength Index)"""
        data = Indicators._check_data(data)
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.DataFrame, fast=12, slow=26, signal=9):
        """Схождение/расхождение скользящих средних (MACD)"""
        data = Indicators._check_data(data)
        ema_fast = Indicators.ema(data, fast)
        ema_slow = Indicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period=20, num_std=2):
        """Полосы Боллинджера (Bollinger Bands)"""
        data = Indicators._check_data(data)

        sma = Indicators.sma(data, period)
        std = data['close'].rolling(period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def bollinger_bands_normalized(data: pd.DataFrame, period=20, num_std=2):
        """Полосы Боллинджера (Bollinger Bands)"""
        data = Indicators._check_data(data)

        if "UpperBB" in data.columns and "MiddleBB" in data.columns and "LowerBB" in data.columns:
            df = data.copy()
        else:
            df = data.copy()
            df['UpperBB'], df['MiddleBB'], df['LowerBB'] = Indicators.bollinger_bands(data, period, num_std)

        upper_norm = (df['UpperBB'] - df['close']) / df['close']
        lower_norm= (df['LowerBB'] - df['close']) / df['close']
        middle_norm = (df['MiddleBB'] - df['close']) / df['close']

        return upper_norm, middle_norm, lower_norm
    
    @staticmethod
    def atr(data: pd.DataFrame, period=14):
        """Average True Range"""
        data = Indicators._check_data(data)
        high_low = data['max'] - data['min']
        high_close = (data['max'] - data['close'].shift()).abs()
        low_close = (data['min'] - data['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def stochastic_oscillator(data: pd.DataFrame, period=14, smoothing=3):
        """Стохастический осциллятор"""
        data = Indicators._check_data(data)
        low_min = data['min'].rolling(period).min()
        high_max = data['max'].rolling(period).max()
        close = data['close']
        
        k = 100 * (close - low_min) / (high_max - low_min)
        d = k.rolling(smoothing).mean()
        return k, d
    
    @staticmethod
    def vwap(data: pd.DataFrame):
        """
        Рассчитывает Volume-Weighted Average Price (VWAP).
        
        Параметры:
            data (DataFrame): DataFrame с колонками ['max', 'min', 'close', 'volume'] и индексом DatetimeIndex.
            
        Возвращает:
            Series: Серия с значениями VWAP.
        """
        data_original = data.copy()
        data = Indicators._check_data(data)

        data.set_index('datetime', inplace=True)
        # Типичная цена
        typical_price = (data['max'] + data['min'] + data['close']) / 3
        
         # Группировка по дням через pd.Grouper (без использования .grouper)
        group_key = pd.Grouper(freq='D')
        
        # Расчет кумулятивных сумм
        cumulative_tpv = (typical_price * data['volume']).groupby(group_key).cumsum()
        cumulative_vol = data['volume'].groupby(group_key).cumsum()
        
        # Расчет VWAP
        vwap = cumulative_tpv / cumulative_vol

        # Восстановление индекса
        vwap.index = data_original.index

        return vwap
    
    @staticmethod
    def vwap_normalized(data: pd.DataFrame):
        data = Indicators._check_data(data)
        df = data.copy()

        if 'VWAP' not in df.columns:
            df['VWAP'] = Indicators.vwap(df)

        vwap_close_ratio = (df['VWAP'] - df['close']) / df['close']
        return vwap_close_ratio
    
    @staticmethod
    def obv(data: pd.DataFrame):
        """
        Рассчитывает On-Balance Volume (OBV).
        
        Параметры:
            data (DataFrame): DataFrame с колонками ['close', 'volume'].
            
        Возвращает:
            Series: Серия с значениями OBV.
        """
        data = Indicators._check_data(data)

        # Расчет направления движения цены
        close_diff = data['close'].diff()
        direction = close_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Расчет OBV
        obv = (direction * data['volume']).cumsum()
        return obv
    
    @staticmethod
    def mfi(data: pd.DataFrame, period=14):
        """
        Рассчитывает Money Flow Index (MFI).
        
        Параметры:
            data (DataFrame): DataFrame с колонками ['max', 'min', 'close', 'volume'].
            period (int): Период расчета (по умолчанию 14).
            
        Возвращает:
            Series: Серия с значениями MFI.
        """
        data = Indicators._check_data(data)
        
        # Расчет типичной цены и денежного потока
        typical_price = (data['max'] + data['min'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        # Определение направления денежного потока
        prev_typical = typical_price.shift(1)
        positive_flow = money_flow.where(typical_price > prev_typical, 0)
        negative_flow = money_flow.where(typical_price < prev_typical, 0)
        
        # Суммирование потоков за период
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        # Расчет коэффициента денежного потока
        mf_ratio = positive_sum / negative_sum
        
        # Расчет MFI с обработкой исключений
        mfi = 100 - (100 / (1 + mf_ratio))
        mfi = mfi.fillna(50)  # Если оба потока равны нулю
        
        return mfi
    
    @staticmethod
    def crv(data: pd.DataFrame, period=20, trading_days=252):
        """
        Рассчитывает реализованную волатильность.
        
        Параметры:
        data: DataFrame с колонками ['close']
        period: период расчета волатильности (дни)
        trading_days: число торговых дней в году (для annualization)
        """
        data = Indicators._check_data(data)

        prices = data['close']
        log_returns = np.log(prices / prices.shift(1))
        daily_vol = log_returns.rolling(period).std()
        annualized_vol = daily_vol * np.sqrt(trading_days)

        return annualized_vol
    
    @classmethod
    def get_shape(cls, indicator_name: str):
        """Получить форму индикатора"""
        
        shape = cls.collumns_shape.get(indicator_name)
        if shape is None:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        elif isinstance(shape, str):
            return 1
        elif isinstance(shape, list):
            return len(shape)
        else:
            raise ValueError(f"Unknown indicator shape: {shape}")
    
    @staticmethod
    def normalize(data: pd.DataFrame, column_indecater:str, column: str = "close"):
        # min_col = data[column_indecater].rolling(period, min_periods=1).min()
        # max_col = data[column_indecater].rolling(period, min_periods=1).max()
        # normalized = (data[column_indecater] - min_col) / (max_col - min_col + 1e-8)
        normalized = (data[column_indecater] - data[column]) / data[column] * 100

        return normalized
    
    @staticmethod
    def paser_collumn_name(collumn_name: str, **kwargs) -> str:
        """Преобразовать имя колонки с учетом периода и торговых дней"""
        if "{period}" in collumn_name:
            collumn_name = collumn_name.replace("{period}", str(kwargs.get('period', 14)))

        if "{trading_days}" in collumn_name:
            collumn_name = collumn_name.replace("{trading_days}", str(kwargs.get('trading_days', 252)))

        return collumn_name
    
    @staticmethod
    def procces_data(collumn_name, data, result, **kwargs):
        if isinstance(collumn_name, str):
            collumn_name = Indicators.paser_collumn_name(collumn_name, **kwargs)

            if collumn_name in data.columns:
                data = data.drop(columns=[collumn_name])

            data[collumn_name] = result
            data[collumn_name] = data[collumn_name].fillna(-1)
        
        elif isinstance(collumn_name, list):
            for i, col in enumerate(collumn_name):
                if col in data.columns:
                    data = data.drop(columns=[col])
                
                data[col] = result[i]
                data[col] = data[col].fillna(-1)
                
        if data.dropna().empty:
            print(result)
            
            raise ValueError(f"Warning: All data is NaN after adding {collumn_name}")

        return data

    @classmethod
    def calculate_normalized(cls, indicator_name: str, data: pd.DataFrame, **kwargs):
        indicators_normalized = {
            'SMA': Indicators.normalize,
            'EMA': Indicators.normalize,
            'BOLLINGER': Indicators.bollinger_bands_normalized,
            'VWAP': Indicators.vwap_normalized,
        }

        if not indicators_normalized.get(indicator_name):
            return data
        
        collumn_name = cls.collumns_shape[indicator_name]
        data = Indicators._check_data(data)

        if indicator_name in ['SMA', 'EMA']:
            collumn_name = collumn_name.replace("{period}", str(kwargs['period']))

            kwargs['column_indecater'] = collumn_name
            kwargs.pop("period", None)

        result = indicators_normalized[indicator_name](data, **kwargs)

        return Indicators.procces_data(collumn_name, data, result, **kwargs)
    
    indicators = {
        'SMA': sma,
        'EMA': ema,
        'RSI': rsi,
        'MACD': macd,
        'BOLLINGER': bollinger_bands,
        'ATR': atr,
        'STOCHASTIC_OSCILLATOR': stochastic_oscillator,
        'VWAP': vwap,
        'OBV': obv,
        "MFI": mfi,
        "CRV": crv
    }

    indicators_input = {
        "SMA": {"period": "int", "column": "str"},
        "EMA": {"period": "int", "column": "str"},
        "BOLLINGER": {"period": "int", "num_std": "int"},
        "VWAP": {},
        "RSI": {"period": "int"},
        "ATR": {"period": "int"},
        "MACD": {"fast": "int", "slow": "int", "signal": "int"},
        "STOCHASTIC_OSCILLATOR": {"period": "int", "smoothing": "int"},
        "OBV": {},
        "MFI": {"period": "int"},
        "CRV": {"period": "int", "trading_days": "int"}
    }

    @classmethod
    def calculate(cls, indicator_name: str, data: pd.DataFrame, **kwargs):
        """Вычислить индикатор"""

        if indicator_name not in cls.indicators:
            return data
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        collumn_name = cls.collumns_shape[indicator_name]
        # data = Indicators._check_data(data)
        result = cls.indicators[indicator_name](data, **kwargs)

        return Indicators.procces_data(collumn_name, data, result, **kwargs)
            
    
# Пример использования:
if __name__ == "__main__":
    from random import randint

    # Загрузка данных из CSV
    data = {
        "datetime": pd.date_range(start="2022-01-01", end="2022-12-31", freq="D"), 
        "open": [randint(1, 100) for _ in range(len(pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")))],
        "max": [randint(1, 100) for _ in range(len(pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")))],
        "min": [randint(1, 100) for _ in range(len(pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")))],
        "close": [randint(1, 100) for _ in range(len(pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")))],
        "volume": [randint(1, 100) for _ in range(len(pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")))]}
    df = pd.DataFrame(data)

    indecaters = {
        "SMA": {"period": "?", "column": "?"},
        "EMA": {"period": "?", "column": "?"},
        "BOLLINGER": {"period": "?", "num_std": "?"},
        "VWAP": {},
        "RSI": {"period": "?"},
        "ATR": {"period": "?"},
        "MACD": {"fast": "?", "slow": "?", "signal": "?"},
        "STOCHASTIC_OSCILLATOR": {"period": "?", "smoothing": "?"},
        "OBV": {},
        "MFI": {"period": "?"},
        "CRV": {"period": "?", "trading_days": 362880}
    }

    def parser_kwargs(kwargs):
        for key, value in kwargs.items():
            if key == "period":
                kwargs[key] = randint(1, 100)
            elif key == "num_std":
                kwargs[key] = randint(1, 100)
            elif key == "fast":
                kwargs[key] = randint(1, 100)
            elif key == "slow":
                kwargs[key] = randint(1, 100)
            elif key == "signal":
                kwargs[key] = randint(1, 100)
            elif key == "smoothing":
                kwargs[key] = randint(1, 100)
            elif key == "trading_days":
                kwargs[key] = randint(1, 100)
            else:
                kwargs[key] = "close"

        return kwargs

    for indecate_name in Indicators.indicators.keys(): 
        kwargs = indecaters[indecate_name]
        kwargs = parser_kwargs(kwargs)
        df = Indicators.calculate(indecate_name, df, **kwargs)
        df = Indicators.calculate_normalized(indecate_name, df, **kwargs)
    
    # # Создание индикаторов
    # indicators = Indicators(df)
    
    # # Расчет индикаторов
    # df['SMA20'] = indicators.sma(20)
    # df['RSI14'] = indicators.rsi()
    # df['MACD'], df['Signal'] = indicators.macd()
    # df['UpperBB'], df['MiddleBB'], df['LowerBB'] = indicators.bollinger_bands()
    # df['ATR14'] = indicators.atr()
    # df['%K'], df['%D'] = indicators.stochastic_oscillator()
    
    # Вывод последних 5 строк
    df = df.tail()
    for collumn in df.columns:
        print(collumn, df[collumn])