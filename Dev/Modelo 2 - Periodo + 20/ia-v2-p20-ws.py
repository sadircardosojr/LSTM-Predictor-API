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
window_size = 72 # Período de tempo para previsão

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

# Usar a primeira coluna numérica como base
sensor = value_cols[0]
df = df[[sensor]].dropna()

# Detectar granularidade
granularidade = df.index.to_series().diff().median()
print(f"Sensor usado: {sensor}")
print(f"Granularidade detectada: {granularidade}")

# === 3. Normalizar ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Criar janelas de entrada/saída
def create_dataset(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


X, y = create_dataset(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Dividir treino/teste
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# === 4. Modelo LSTM ===
model = Sequential([
    LSTM(50, input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

# === 5. Previsão de teste ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# === 6. Projeção futura ===
janela_atual = scaled_data[-window_size:].reshape(1, window_size, 1)
futuro_escalado = []

for _ in range(n_periodos):
    pred = model.predict(janela_atual)[0][0]  # valor escalar
    futuro_escalado.append(pred)

    nova_janela = np.append(janela_atual[0, 1:, 0], [pred])  # shape (window_size,)
    janela_atual = nova_janela.reshape(1, window_size, 1)

futuro = scaler.inverse_transform(np.array(futuro_escalado).reshape(-1, 1))
datas_futuras = [df.index[-1] + granularidade * (i+1) for i in range(n_periodos)]

# === 7. Plot e salvar imagem ===
plt.figure(figsize=(14,6))
plt.plot(df.index[-len(y_true):], y_true, label='Real', color='blue')
plt.plot(df.index[-len(y_true):], y_pred, label='Previsto (teste)', color='orange')
plt.plot(datas_futuras, futuro, label='Futuro projetado', color='green', linestyle='--', marker='o')
plt.title(f'Previsão com LSTM - {sensor}')
plt.xlabel('Tempo')
plt.ylabel(sensor)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('previsao_lstm_futuro.png')
plt.close()

print("✅ Previsão gerada com sucesso.")
print("📁 Gráfico salvo como previsao_lstm_futuro.png")
print(f"📉 Erro MSE no teste: {mean_squared_error(y_true, y_pred):.6f}")
