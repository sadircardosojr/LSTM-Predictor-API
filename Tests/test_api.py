import pytest
import json
import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path para importar módulos
sys.path.append(str(Path(__file__).parent.parent / '_Docker'))

from app import app
from ia_model import LSTMPredictor

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint_valid_data(client):
    """Teste do endpoint /predict com dados válidos"""
    data = {
        "n_periodos_compar": 20,
        "taxa_de_analise": 0.55,
        "data": [
            {"__time": 1625097600000, "dim": 1, "valor1": 10, "valor2": 20},
            {"__time": 1625184000000, "dim": 1, "valor1": 11, "valor2": 21},
            {"__time": 1625270400000, "dim": 1, "valor1": 12, "valor2": 22},
            {"__time": 1625356800000, "dim": 1, "valor1": 13, "valor2": 23},
            {"__time": 1625443200000, "dim": 1, "valor1": 14, "valor2": 24}
        ]
    }
    
    response = client.post('/predict',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert 'predictions' in response_data
    assert 'metrics' in response_data

def test_predict_endpoint_invalid_data(client):
    """Teste do endpoint /predict com dados inválidos"""
    data = {
        "n_periodos_compar": 20,
        "taxa_de_analise": 0.55,
        "data": []  # Lista vazia deve gerar erro
    }
    
    response = client.post('/predict',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 400

def test_predict_endpoint_missing_fields(client):
    """Teste do endpoint /predict com campos faltando"""
    data = {
        "data": [
            {"__time": 1625097600000, "dim": 1, "valor1": 10, "valor2": 20}
        ]
    }
    
    response = client.post('/predict',
                         data=json.dumps(data),
                         content_type='application/json')
    
    assert response.status_code == 400

def test_lstm_predictor_initialization():
    """Teste de inicialização do LSTMPredictor"""
    predictor = LSTMPredictor(n_periodos=20, taxa_de_analise=0.55)
    assert predictor.n_periodos == 20
    assert predictor.taxa_de_analise == 0.55
    assert predictor.scaler is not None
    assert predictor.model is None

def test_detect_time_column():
    """Teste da função detect_time_column"""
    import pandas as pd
    predictor = LSTMPredictor()
    
    # Criar DataFrame de teste
    df = pd.DataFrame({
        '__time': [1625097600000, 1625184000000],
        'valor1': [10, 11]
    })
    
    df_processed, time_col = predictor.detect_time_column(df)
    assert time_col == '__time'
    assert pd.api.types.is_datetime64_any_dtype(df_processed.index)

if __name__ == '__main__':
    pytest.main([__file__]) 