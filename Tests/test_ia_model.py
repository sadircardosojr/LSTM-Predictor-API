import pytest
import pandas as pd
import numpy as np
from ia_model import LSTMPredictor

def test_detect_time_column(sample_dataframe):
    """Testa a detecção da coluna de tempo."""
    predictor = LSTMPredictor()
    time_col = predictor.detect_time_column(sample_dataframe)
    assert time_col == "__time"

def test_create_dataset():
    """Testa a criação do dataset para treinamento."""
    predictor = LSTMPredictor()
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window_size = 3
    
    X, y = predictor.create_dataset(data, window_size)
    
    assert X.shape == (len(data) - window_size, window_size)
    assert y.shape == (len(data) - window_size,)
    assert np.array_equal(X[0], [1, 2, 3])
    assert y[0] == 4

def test_process_and_predict(sample_data):
    """Testa o processamento e predição completos."""
    predictor = LSTMPredictor()
    result = predictor.process_and_predict(sample_data)
    
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "metrics" in result
    assert isinstance(result["predictions"], list)
    assert isinstance(result["metrics"], dict)

def test_invalid_input():
    """Testa o comportamento com entrada inválida."""
    predictor = LSTMPredictor()
    with pytest.raises(ValueError):
        predictor.process_and_predict({"data": []})

def test_data_normalization(sample_dataframe):
    """Testa a normalização dos dados."""
    predictor = LSTMPredictor()
    normalized_data = predictor.normalize_data(sample_dataframe[["valor1", "valor2"]].values)
    
    assert normalized_data.shape == sample_dataframe[["valor1", "valor2"]].values.shape
    assert np.all(normalized_data >= -1) and np.all(normalized_data <= 1) 