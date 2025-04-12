import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === CONFIGURAÇÃO ===
arquivo_json = "base.json"
n_periodos = 20 # Número de períodos futuros a projetar
TAXA_DE_ANALISE = 0.45

# === 1. Carregar dados ===
with open(arquivo_json, "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# === 2. Detectar colunas ===
# Forçar identificação de coluna de tempo
if "__time" in df.columns:
    df["__time"] = pd.to_datetime(df["__time"], unit="ms")
    df.set_index("__time", inplace=True)

# Detectar colunas numéricas válidas
value_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and df[col].notna().sum() > 0.9 * len(df)]

if not value_cols:
    raise ValueError("Nenhuma coluna numérica válida foi encontrada.")

# Usar todas as colunas numéricas
print(f"Sensores usados: {value_cols}")
df = df[value_cols].dropna()

# Calcular window_size dinamicamente com validações
total_periods = len(df)
max_window_size = 1000  # Definir um limite máximo se necessário
min_window_size = 24    # Definir um limite mínimo (ex: 1 dia se dados horários)

# Calcular window_size como 80% dos dados, respeitando os limites
window_size = int(total_periods * TAXA_DE_ANALISE)
window_size = min(window_size, max_window_size)  # Não exceder o máximo
window_size = max(window_size, min_window_size)  # Não ficar abaixo do mínimo

print(f"Períodos totais disponíveis: {total_periods}")
print(f"Tamanho da janela de análise: {window_size} períodos")

# Detectar granularidade
granularidade = df.index.to_series().diff().median()
print(f"Granularidade detectada: {granularidade}")

# === 3. Normalizar ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Modificar a função create_dataset para trabalhar com múltiplas variáveis
def create_dataset(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, window_size)
# Ajustar o reshape para considerar múltiplas features
X = X.reshape((X.shape[0], X.shape[1], len(value_cols)))

# Dividir treino/teste
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# === 4. Modelo LSTM ===
model = Sequential([
    LSTM(100, input_shape=(window_size, len(value_cols))),
    Dense(len(value_cols))
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

# === 5. Previsão de teste ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# === 6. Projeção futura ===
janela_atual = scaled_data[-window_size:].reshape(1, window_size, len(value_cols))
futuro_escalado = []

for _ in range(n_periodos):
    pred = model.predict(janela_atual)[0]  # agora retorna um vetor
    futuro_escalado.append(pred)
    
    nova_janela = np.append(janela_atual[0, 1:, :], [pred], axis=0)
    janela_atual = nova_janela.reshape(1, window_size, len(value_cols))

futuro = scaler.inverse_transform(np.array(futuro_escalado))
datas_futuras = [df.index[-1] + granularidade * (i+1) for i in range(n_periodos)]

# === 7. Plot e salvar imagem ===
plt.figure(figsize=(14,6))
for i, col in enumerate(value_cols):
    plt.plot(df.index[-len(y_true):], y_true[:, i], label=f'Real ({col})', alpha=0.7)
    plt.plot(df.index[-len(y_true):], y_pred[:, i], label=f'Previsto ({col})', linestyle='--', alpha=0.7)
    plt.plot(datas_futuras, futuro[:, i], label=f'Futuro ({col})', linestyle=':', marker='o', alpha=0.7)

plt.title('Previsão com LSTM - Múltiplas Variáveis')
plt.xlabel('Tempo')
plt.ylabel('Valores')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('previsao_lstm_futuro_multivariavel.png')
plt.close()

print("✅ Previsão gerada com sucesso.")
print("📁 Gráfico salvo como previsao_lstm_futuro_multivariavel.png")
print(f"📉 Erro MSE no teste: {mean_squared_error(y_true, y_pred):.6f}")
