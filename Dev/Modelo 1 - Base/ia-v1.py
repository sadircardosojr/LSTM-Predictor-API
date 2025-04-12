import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json

# === 1. Carregar dados ===
with open("base.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
df['__time'] = pd.to_datetime(df['__time'], unit='ms')
df.set_index('__time', inplace=True)

# === 2. Pré-processamento ===
sensor = 'BATCS_EPCVMH_SMF01'
df = df[[sensor]].dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_dataset(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_dataset(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# === 3. Modelo LSTM ===
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# === 4. Previsão ===
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_true = scaler.inverse_transform(y_test)

# === 5. Gerar imagem ===
plt.figure(figsize=(14,6))
plt.plot(df.index[-len(y_true):], y_true, label='Real', color='blue')
plt.plot(df.index[-len(y_true):], predicted, label='Previsto', color='orange')
plt.title(f'Previsão com LSTM - Sensor: {sensor}')
plt.xlabel('Tempo')
plt.ylabel(sensor)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('previsao_lstm.png')  # Salva o gráfico como imagem
plt.close()

# === 6. Mostrar erro ===
mse = mean_squared_error(y_true, predicted)
print(f"Mean Squared Error: {mse:.6f}")
print("Gráfico salvo como previsao_lstm.png")
