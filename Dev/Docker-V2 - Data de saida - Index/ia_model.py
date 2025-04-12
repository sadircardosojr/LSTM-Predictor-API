import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMPredictor:
    def __init__(self, n_periodos=20, taxa_de_analise=0.55):
        self.n_periodos = n_periodos
        self.taxa_de_analise = taxa_de_analise
        self.scaler = MinMaxScaler()
        self.model = None
        self.value_cols = None
        
    def detect_time_column(self, df):
        """
        Detecta e converte a coluna de tempo no DataFrame.
        """
        # Lista de possíveis nomes de colunas de tempo
        time_column_names = ['time', '__time', 'timestamp', 'date', 'datetime', 'created_at', 'updated_at']
        
        # Primeiro, procura por nomes comuns
        for col in df.columns:
            if str(col).lower() in time_column_names:
                try:
                    # Tenta converter de millisegundos epoch
                    df[col] = pd.to_datetime(df[col], unit='ms')
                    print(f"✅ Coluna de tempo detectada e convertida: {col} (formato: millisegundos)")
                    return df, col
                except:
                    try:
                        # Tenta converter como string de data/hora
                        df[col] = pd.to_datetime(df[col])
                        print(f"✅ Coluna de tempo detectada e convertida: {col} (formato: data/hora)")
                        return df, col
                    except Exception as e:
                        print(f"❌ Erro ao converter coluna {col}: {str(e)}")
                        continue
        
        # Se não encontrou pelos nomes comuns, procura por colunas que parecem conter tempo
        for col in df.columns:
            # Verifica se a coluna tem números grandes (possíveis timestamps)
            if df[col].dtype in ['int64', 'float64']:
                if df[col].min() > 1000000000000:  # Timestamp em millisegundos (após 2001)
                    try:
                        df[col] = pd.to_datetime(df[col], unit='ms')
                        print(f"✅ Coluna de tempo detectada e convertida: {col} (formato: millisegundos)")
                        return df, col
                    except Exception as e:
                        print(f"❌ Erro ao converter coluna {col}: {str(e)}")
                elif df[col].min() > 1000000000:  # Timestamp em segundos (após 2001)
                    try:
                        df[col] = pd.to_datetime(df[col], unit='s')
                        print(f"✅ Coluna de tempo detectada e convertida: {col} (formato: segundos)")
                        return df, col
                    except Exception as e:
                        print(f"❌ Erro ao converter coluna {col}: {str(e)}")
        
        # Se não encontrou coluna de tempo, cria uma artificial
        print("✅ Nenhuma coluna de tempo válida encontrada. Criando índice artificial...")
        df.index = pd.date_range(start='now', periods=len(df), freq='H')
        return df, None
    
    def create_dataset(self, data, window_size=24):
        """
        Cria conjuntos de dados para treinamento do modelo LSTM.
        
        Args:
            data: Array numpy com os dados normalizados
            window_size: Tamanho da janela para previsão
            
        Returns:
            X: Array com as janelas de entrada
            y: Array com os valores alvo
        """
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
    
    def tratar_nan_com_media_adjacente(self, df, colunas):
        """
        Trata valores NaN usando a média entre o valor anterior e o próximo.
        Se não houver valor anterior ou próximo, usa o valor mais próximo disponível.
        """
        for col in colunas:
            # Identifica índices com NaN
            nan_indices = df[col].isna()
            if not nan_indices.any():
                continue
            
            # Para cada NaN, encontra o valor anterior e próximo não-NaN
            for idx in df[nan_indices].index:
                # Encontra o valor anterior não-NaN
                valor_anterior = None
                for i in range(1, len(df)):
                    idx_anterior = df.index[df.index.get_loc(idx) - i]
                    if idx_anterior in df.index and not pd.isna(df.loc[idx_anterior, col]):
                        valor_anterior = df.loc[idx_anterior, col]
                        break
                    
                # Encontra o próximo valor não-NaN
                valor_proximo = None
                for i in range(1, len(df)):
                    idx_proximo = df.index[df.index.get_loc(idx) + i]
                    if idx_proximo in df.index and not pd.isna(df.loc[idx_proximo, col]):
                        valor_proximo = df.loc[idx_proximo, col]
                        break
                    
                # Calcula a média ou usa o valor disponível
                if valor_anterior is not None and valor_proximo is not None:
                    df.loc[idx, col] = (valor_anterior + valor_proximo) / 2
                elif valor_anterior is not None:
                    df.loc[idx, col] = valor_anterior
                elif valor_proximo is not None:
                    df.loc[idx, col] = valor_proximo
                else:
                    # Se não houver valores adjacentes, usa a média geral da coluna
                    df.loc[idx, col] = df[col].mean()
                
        return df

    def process_and_predict(self, data):
        try:
            # Converter dados para DataFrame
            df = pd.DataFrame(data)
            print(f"✅ Colunas recebidas: {df.columns.tolist()}")
            
            # Verificar volume mínimo de registros
            if len(df) < 10:
                raise ValueError(f"❌ É necessário pelo menos 10 registros para análise. Foram fornecidos apenas {len(df)} registros.")
            
            # Processar coluna de tempo
            df, time_column = self.detect_time_column(df)
            
            if time_column is not None:
                df.set_index(time_column, inplace=True)
                print(f"✅ Usando coluna '{time_column}' como índice temporal")
            
            # Ordenar e limpar dados
            if not df.index.is_monotonic_increasing:
                print("✅ Ordenando dados temporalmente...")
                df.sort_index(inplace=True)
            
            if df.index.duplicated().any():
                print("✅ Removendo duplicatas...")
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
            
            print(f"✅ Colunas numéricas detectadas: {self.value_cols}")
            
            # Tratar valores NaN
            print("✅ Tratando valores NaN...")
            df = self.tratar_nan_com_media_adjacente(df, self.value_cols)
            
            # Configurar window_size
            total_periods = len(df)
            window_size = int(total_periods * self.taxa_de_analise)
            window_size = min(max(window_size, 24), 1000)
            
            # Verificar se temos dados suficientes para a janela
            if total_periods <= window_size:
                window_size = max(1, total_periods - 1)
                print(f"✅ Ajustando tamanho da janela para {window_size} devido ao volume limitado de dados")
            
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
            print("✅ Treinando modelo LSTM...")
            self.model.fit(X, y, epochs=30, batch_size=32, verbose=0)
            print("✅ Modelo treinado com sucesso!")
            
            # Fazer previsões futuras
            print("✅ Gerando previsões futuras...")
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
            
            print("✅ Análise concluída com sucesso!")
            return result_df, mse_errors

        except Exception as e:
            print(f"❌ Erro durante o processamento: {str(e)}")
            raise 