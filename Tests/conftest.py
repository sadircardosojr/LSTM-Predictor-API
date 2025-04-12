import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
    """Fixture que retorna dados de exemplo para testes."""
    dates = [datetime.now() + timedelta(days=i) for i in range(10)]
    data = {
        "__time": [d.timestamp() * 1000 for d in dates],
        "dim": [1] * 10,
        "valor1": np.random.rand(10) * 100,
        "valor2": np.random.rand(10) * 100
    }
    return {
        "n_periodos_compar": 5,
        "taxa_de_analise": 0.55,
        "data": [
            {key: value[i] for key, value in data.items()}
            for i in range(len(data["__time"]))
        ]
    }

@pytest.fixture
def sample_dataframe():
    """Fixture que retorna um DataFrame de exemplo para testes."""
    dates = [datetime.now() + timedelta(days=i) for i in range(10)]
    return pd.DataFrame({
        "__time": dates,
        "dim": [1] * 10,
        "valor1": np.random.rand(10) * 100,
        "valor2": np.random.rand(10) * 100
    }) 