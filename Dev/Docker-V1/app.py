from flask import Flask, request, jsonify
from ia_model import LSTMPredictor
import pandas as pd
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber dados do request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type deve ser application/json'
            }), 415

        content = request.get_json()
        
        # Validar dados de entrada
        required_fields = ['n_periodos_compar', 'taxa_de_analise', 'data']
        missing_fields = [field for field in required_fields if field not in content]
        if missing_fields:
            return jsonify({
                'error': f'Campos obrigatórios ausentes: {", ".join(missing_fields)}'
            }), 400

        # Validar se data é uma lista não vazia
        if not isinstance(content['data'], list) or not content['data']:
            return jsonify({
                'error': 'O campo "data" deve ser uma lista não vazia'
            }), 400

        
        # Instanciar o preditor
        predictor = LSTMPredictor(
            n_periodos=content['n_periodos_compar'],
            taxa_de_analise=content['taxa_de_analise']
        )
        
        # Processar dados e fazer previsão
        result_df, mse_errors = predictor.process_and_predict(content['data'])
        
        # Converter resultado para o formato desejado
        result_data = result_df.reset_index().to_dict('records')
        
        return jsonify({
            'taxa_de_erros_por_dimensao': mse_errors,
            'data': result_data
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Log completo do erro
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 