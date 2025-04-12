import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import os
import sys
import logging
import tensorflow as tf

# Configurar logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configurar TensorFlow para usar CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

class LSTMPredictor:
    def __init__(self, n_periodos=20, taxa_de_analise=0.55):
        self.n_periodos = n_periodos
        self.taxa_de_analise = taxa_de_analise
        self.scaler = MinMaxScaler()
        self.model = None
        self.value_cols = None
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"✅ Inicializando LSTMPredictor com {self.n_jobs} threads")
        
    def detect_time_column(self, df):
        """
        Detecta e converte a coluna de tempo no DataFrame.
        """
        logger.info("🔍 Iniciando detecção de coluna temporal...")
        
        # Lista de possíveis nomes de colunas de tempo
        time_column_names = ['time', '__time', 'timestamp', 'date', 'datetime', 'created_at', 'updated_at']
        
        # Primeiro, procura por nomes comuns
        for col in df.columns:
            if str(col).lower() in time_column_names:
                try:
                    # Tenta converter de millisegundos epoch
                    df[col] = pd.to_datetime(df[col], unit='ms')
                    logger.info(f"✅ Coluna de tempo detectada e convertida: {col} (formato: millisegundos)")
                    return df, col
                except:
                    try:
                        # Tenta converter como string de data/hora
                        df[col] = pd.to_datetime(df[col])
                        logger.info(f"✅ Coluna de tempo detectada e convertida: {col} (formato: data/hora)")
                        return df, col
                    except Exception as e:
                        logger.error(f"❌ Erro ao converter coluna {col}: {str(e)}")
                        continue
        
        # Se não encontrou pelos nomes comuns, procura por colunas que parecem conter tempo
        for col in df.columns:
            logger.debug(f"Analisando coluna: {col}")
            # Verifica se a coluna tem números grandes (possíveis timestamps)
            if df[col].dtype in ['int64', 'float64']:
                if df[col].min() > 1000000000000:  # Timestamp em millisegundos (após 2001)
                    try:
                        df[col] = pd.to_datetime(df[col], unit='ms')
                        logger.info(f"✅ Coluna de tempo detectada e convertida: {col} (formato: millisegundos)")
                        return df, col
                    except Exception as e:
                        logger.error(f"❌ Erro ao converter coluna {col}: {str(e)}")
                elif df[col].min() > 1000000000:  # Timestamp em segundos (após 2001)
                    try:
                        df[col] = pd.to_datetime(df[col], unit='s')
                        logger.info(f"✅ Coluna de tempo detectada e convertida: {col} (formato: segundos)")
                        return df, col
                    except Exception as e:
                        logger.error(f"❌ Erro ao converter coluna {col}: {str(e)}")
        
        # Se não encontrou coluna de tempo, cria uma artificial
        logger.info("✅ Nenhuma coluna de tempo válida encontrada. Criando índice artificial...")
        granularity = self.detect_granularity(df)
        df.index = pd.date_range(start='now', periods=len(df), freq=granularity)
        return df, None
    
    def detect_granularity(self, df):
        """
        Detecta a granularidade dos dados com base na diferença entre as datas consecutivas.
        """
        if len(df) < 2:
            return 'H'  # Default para uma hora se não houver dados suficientes

        # Calcula a diferença entre as datas consecutivas
        time_diff = df.index[1] - df.index[0]

        # Determina a granularidade com base na diferença
        if time_diff.days > 0:
            return 'D'  # Diária
        elif time_diff.seconds >= 3600:
            return 'H'  # Horária
        elif time_diff.seconds >= 60:
            return 'T'  # Minutos
        else:
            return 'S'  # Segundos
    
    def create_dataset(self, data, window_size=24):
        """
        Cria conjuntos de dados para treinamento do modelo LSTM.
        Versão otimizada usando vetorização.
        """
        # Usar vetorização em vez de loops para melhor desempenho
        n_samples = len(data) - window_size
        X = np.array([data[i:i+window_size] for i in range(n_samples)])
        y = np.array([data[i+window_size] for i in range(n_samples)])
        return X, y
    
    def tratar_nan_com_media_adjacente(self, df, colunas):
        """
        Trata valores NaN usando a média entre o valor anterior e o próximo.
        Versão otimizada usando operações vetorizadas.
        """
        for col in colunas:
            # Identifica índices com NaN
            nan_indices = df[col].isna()
            if not nan_indices.any():
                continue
            
            # Usar método de interpolação do pandas para preencher NaN
            df[col] = df[col].interpolate(method='linear')
            
            # Se ainda houver NaN (no início ou fim), preencher com o valor mais próximo
            if df[col].isna().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
        return df

    def process_and_predict(self, data):
        try:
            start_time = time.time()
            logger.info("🚀 Iniciando processamento e predição...")
            
            # Converter dados para DataFrame
            df = pd.DataFrame(data)
            logger.info(f"✅ Dados recebidos. Shape: {df.shape}")
            logger.info(f"✅ Colunas: {df.columns.tolist()}")
            
            # Verificar volume mínimo de registros
            if len(df) < 10:
                msg = f"❌ É necessário pelo menos 10 registros para análise. Foram fornecidos apenas {len(df)} registros."
                logger.error(msg)
                raise ValueError(msg)
            
            # Processar coluna de tempo
            df, time_column = self.detect_time_column(df)
            
            if time_column is not None:
                df.set_index(time_column, inplace=True)
                logger.info(f"✅ Usando coluna '{time_column}' como índice temporal")
            
            # Ordenar e limpar dados
            if not df.index.is_monotonic_increasing:
                logger.info("✅ Ordenando dados temporalmente...")
                df.sort_index(inplace=True)
            
            if df.index.duplicated().any():
                logger.info("✅ Removendo duplicatas...")
                df = df[~df.index.duplicated(keep='last')]
            
            # Verificar novamente o volume após limpeza
            if len(df) < 10:
                raise ValueError(f"❌ Após remover duplicatas, restaram apenas {len(df)} registros. É necessário pelo menos 10 registros para análise.")
            
            # Detectar colunas numéricas
            self.value_cols = [col for col in df.columns 
                             if df[col].dtype in ['float64', 'int64'] 
                             and df[col].notna().sum() > 0.9 * len(df)]
            
            if not self.value_cols:
                raise ValueError("❌ Nenhuma coluna numérica válida encontrada nos dados")
            
            logger.info(f"✅ Colunas numéricas detectadas: {self.value_cols}")
            
            # Tratar valores NaN
            logger.info("✅ Tratando valores NaN...")
            df = self.tratar_nan_com_media_adjacente(df, self.value_cols)
            
            # Configurar window_size
            total_periods = len(df)
            window_size = int(total_periods * self.taxa_de_analise)
            window_size = min(max(window_size, 24), 1000)
            
            # Verificar se temos dados suficientes para a janela
            if total_periods <= window_size:
                window_size = max(1, total_periods - 1)
                logger.info(f"✅ Ajustando tamanho da janela para {window_size} devido ao volume limitado de dados")
            
            # Preparar dados
            scaled_data = self.scaler.fit_transform(df[self.value_cols])
            X, y = self.create_dataset(scaled_data, window_size)
            
            # Verificar se temos dados suficientes para treinar
            if len(X) == 0:
                raise ValueError(f"❌ Não há dados suficientes para treinar o modelo. Necessário pelo menos {window_size + 1} registros.")
            
            # Reshape apenas se tivermos dados suficientes
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], X.shape[1], len(self.value_cols)))
            
            # Treinar modelo
            self.model = Sequential([
                LSTM(100, input_shape=(window_size, len(self.value_cols))),
                Dense(len(self.value_cols))
            ])
            self.model.compile(optimizer='adam', loss='mse')
            logger.info("✅ Treinando modelo LSTM...")
            
            # Usar batch_size menor e mais epochs
            self.model.fit(X, y, epochs=20, batch_size=16, verbose=1)
            logger.info("✅ Modelo treinado com sucesso!")
            
            # Fazer previsões
            logger.info("🔮 Gerando previsões...")
            
            # Fazer previsões futuras
            janela_atual = scaled_data[-window_size:].reshape(1, window_size, len(self.value_cols))
            futuro_escalado = []
            
            for _ in range(self.n_periodos):
                pred = self.model.predict(janela_atual, verbose=0)[0]
                futuro_escalado.append(pred)
                nova_janela = np.append(janela_atual[0, 1:, :], [pred], axis=0)
                janela_atual = nova_janela.reshape(1, window_size, len(self.value_cols))
            
            # Processar resultados
            futuro = self.scaler.inverse_transform(np.array(futuro_escalado))
            datas_futuras = [df.index[-1] + pd.Timedelta(hours=i+1) for i in range(self.n_periodos)]
            
            # Criar DataFrame com resultados
            df_futuro = pd.DataFrame(futuro, index=datas_futuras, columns=self.value_cols)
            df_futuro['tipo'] = 'projetado'
            df['tipo'] = 'original'
            
            # Combinar dados originais e projetados
            result_df = pd.concat([df, df_futuro])
            
            # Formatar o índice para o formato americano
            result_df.index = result_df.index.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calcular MSE
            mse_errors = {}
            for i, col in enumerate(self.value_cols):
                mse = mean_squared_error(df[col].values[-self.n_periodos:], 
                                       df_futuro[col].values[:self.n_periodos])
                mse_errors[col] = float(mse)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Análise concluída com sucesso! Tempo de execução: {elapsed_time:.2f} segundos")
            return result_df, mse_errors

        except Exception as e:
            logger.error(f"❌ Erro durante o processamento: {str(e)}", exc_info=True)
            raise
