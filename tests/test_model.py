import pytest
import torch
from torch import nn

from backend.MMM.models.model_trade import (  # Предполагается, что модели находятся в файле models.py
    OHLCV_MLP,
    OHLCV_LSTM,
    OHLCV_TCNN,
    OHLCV_Transformer,
    OHLCV_CNN_LSTM,
    TabTransformer,
    TransformerEncoder,
    PositionalEncoding
)

# Фикстуры для тестовых данных
@pytest.fixture(params=[1, 16])  # Разные размеры батча
def batch_size(request):
    return request.param

@pytest.fixture
def input_seq_data(batch_size):
    # Данные для моделей с последовательностями: [batch, seq_len, input_size]
    seq_len = 30
    input_size = 5  # OHLCV
    return torch.randn(batch_size, seq_len, input_size)

@pytest.fixture
def input_tabtransformer_data(batch_size):
    # Данные для TabTransformer: числовые и категориальные признаки
    num_features = 10
    cat_dims = [5, 3, 7]  # Пример размерностей категориальных признаков
    x_num = torch.randn(batch_size, num_features)
    x_cat = torch.cat([
        torch.randint(0, dim, (batch_size, 1)) for dim in cat_dims
    ], dim=1)
    return x_num, x_cat

# Тесты для каждой модели
def test_mlp(input_seq_data):
    batch_size, seq_len, input_size = input_seq_data.shape
    model = OHLCV_MLP(input_size, seq_len)
    output = model(input_seq_data)
    assert output.shape == (batch_size, 3)

def test_lstm(input_seq_data):
    batch_size, seq_len, input_size = input_seq_data.shape
    model = OHLCV_LSTM(input_size)
    output = model(input_seq_data)
    assert output.shape == (batch_size, 3)

def test_tcnn(input_seq_data):
    batch_size, seq_len, input_size = input_seq_data.shape
    model = OHLCV_TCNN(input_size, seq_len)
    output = model(input_seq_data)
    assert output.shape == (batch_size, 3)

def test_transformer(input_seq_data):
    batch_size, seq_len, input_size = input_seq_data.shape
    model = OHLCV_Transformer(input_size)
    output = model(input_seq_data)
    assert output.shape == (batch_size, 3)

def test_cnn_lstm(input_seq_data):
    batch_size, seq_len, input_size = input_seq_data.shape
    model = OHLCV_CNN_LSTM(input_size)
    output = model(input_seq_data)
    assert output.shape == (batch_size, 3)

def test_tabtransformer(input_tabtransformer_data):
    x_num, x_cat = input_tabtransformer_data
    num_features = x_num.shape[1]
    cat_dims = [5, 3, 7]
    model = TabTransformer(num_features, len(cat_dims), cat_dims)
    output = model(x_num, x_cat)
    assert output.shape == (x_num.shape[0], 3)

# Дополнительные тесты для PositionalEncoding
def test_positional_encoding():
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    x = torch.randn(10, max_len, d_model)
    output = pe(x)
    assert output.shape == x.shape
    # Проверка, что кодирование добавляется, а не заменяет
    assert not torch.allclose(output, x)

# Проверка наличия ожидаемых слоев (опционально)
def test_lstm_structure():
    model = OHLCV_LSTM(input_size=5)
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.fc, nn.Linear)

def test_transformer_structure():
    model = OHLCV_Transformer(input_size=5)
    assert isinstance(model.pos_encoder, PositionalEncoding)
    assert isinstance(model.transformer, TransformerEncoder)
    assert isinstance(model.fc, nn.Linear)