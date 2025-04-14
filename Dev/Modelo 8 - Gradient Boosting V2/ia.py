#%%
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# === CONFIGURAÇÕES ===
arquivo_json = "base.json"
n_periodos = 20
TAXA_DE_ANALISE = 1
PLOT_SOMENTE_ULTIMOS_PERIODOS = 150

def detect_time_column(df):
    time_column_names = ['time', '__time', 'timestamp', 'date', 'datetime', 'created_at', 'updated_at']
    for col in time_column_names:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], unit='ms')
                print(f"Coluna de tempo detectada e convertida: {col} (formato: millisegundos)")
                return df, col
            except:
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Coluna de tempo detectada e convertida: {col} (formato: data/hora)")
                    return df, col
                except:
                    continue
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].min() > 1000000000000:
                try:
                    df[col] = pd.to_datetime(df[col], unit='ms')
                    print(f"Coluna de tempo detectada e convertida: {col} (formato: millisegundos)")
                    return df, col
                except:
                    continue
            elif df[col].min() > 1000000000:
                try:
                    df[col] = pd.to_datetime(df[col], unit='s')
                    print(f"Coluna de tempo detectada e convertida: {col} (formato: segundos)")
                    return df, col
                except:
                    continue
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Coluna de tempo detectada e convertida: {col} (formato: data/hora)")
                return df, col
            except:
                continue
    raise ValueError("Nenhuma coluna de tempo válida foi encontrada no DataFrame")

# === 1. Carregar dados ===
with open(arquivo_json, "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# === 2. Detectar e processar coluna de tempo ===
try:
    df, time_column = detect_time_column(df)
    df.set_index(time_column, inplace=True)
    print(f"Usando coluna '{time_column}' como índice temporal")
except ValueError as e:
    print(f"⚠️ Aviso: {e}")
    df.index = pd.date_range(start='now', periods=len(df), freq='H')
    print("Índice temporal artificial criado com sucesso")

if not df.index.is_monotonic_increasing:
    print("⚠️ Ordenando dados temporais...")
    df.sort_index(inplace=True)

if df.index.duplicated().any():
    df = df[~df.index.duplicated(keep='last')]

value_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and df[col].notna().sum() > 0.9 * len(df)]
if not value_cols:
    raise ValueError("Nenhuma coluna numérica válida foi encontrada.")

print(f"Sensores usados: {value_cols}")
df = df[value_cols].dropna()

# === 3. Criar features derivadas ===
for col in value_cols:
    df[f"{col}_diff1"] = df[col].diff()
    df[f"{col}_ma3"] = df[col].rolling(3).mean()
    df[f"{col}_ma5"] = df[col].rolling(5).mean()

df.dropna(inplace=True)
feature_cols = df.columns.tolist()

# === 4. Janela de entrada ===
total_periods = len(df)
max_window_size = 1000
min_window_size = 24
window_size = int(total_periods * TAXA_DE_ANALISE)
window_size = min(max(window_size, min_window_size), max_window_size)
print(f"Tamanho da janela: {window_size}")

granularidade = df.index.to_series().diff().median()
print(f"Granularidade detectada: {granularidade}")

# === 5. Normalização ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_dataset(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size].flatten())
        y.append(data[i+window_size][:len(value_cols)])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, window_size)
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# === 6. Modelo
base_model = HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=300,
    max_depth=7,
    learning_rate=0.05,
    random_state=42
)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# === 7. Previsão histórica
previsoes_historico_scaled = model.predict(X)
previsoes_historico = scaler.inverse_transform(
    np.hstack([
        previsoes_historico_scaled,
        np.zeros((len(previsoes_historico_scaled), len(feature_cols) - len(value_cols)))
    ])
)[:, :len(value_cols)]

indices_historico = df.index[window_size:]

# === 8. Projeção futura
janela_atual = scaled_data[-window_size:].copy()
futuro_escalado = []

for _ in range(n_periodos):
    janela_input = janela_atual.flatten().reshape(1, -1)
    pred = model.predict(janela_input)[0]
    futuro_escalado.append(pred)

    linha_futura = np.zeros(scaled_data.shape[1])
    linha_futura[:len(value_cols)] = pred
    janela_atual = np.vstack([janela_atual[1:], linha_futura])

futuro = scaler.inverse_transform(
    np.hstack([
        futuro_escalado,
        np.zeros((n_periodos, len(feature_cols) - len(value_cols)))
    ])
)[:, :len(value_cols)]

datas_futuras = [df.index[-1] + granularidade * (i + 1) for i in range(n_periodos)]

# === 9. Plot ===
plt.figure(figsize=(20,10))

if PLOT_SOMENTE_ULTIMOS_PERIODOS:
    tempo_minimo = df.index[-1] - PLOT_SOMENTE_ULTIMOS_PERIODOS * granularidade
    df_plot = df[df.index >= tempo_minimo]
else:
    df_plot = df

for i, col in enumerate(value_cols):
    plt.plot(df_plot.index, df_plot[col].values, label=f'Histórico Real ({col})', alpha=0.6, linewidth=1.5)
    plt.plot(datas_futuras, futuro[:, i], label=f'Projeção ({col})', linestyle='--', marker='o', linewidth=2)

plt.title('Histórico + Projeção Gradient Boosting com Features Derivadas')
plt.xlabel('Tempo')
plt.ylabel('Valores')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

data_hora_atual = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_arquivo = f"previsao_gb_derivadas_{data_hora_atual}.png"
plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
plt.close()

print(f"📁 Gráfico salvo como {nome_arquivo}")

# === 10. Erro MSE
mse_historico = {}
for i, col in enumerate(value_cols):
    mse = mean_squared_error(df[col].values[window_size:], previsoes_historico[:, i])
    mse_historico[col] = mse

print("✅ Análise com features derivadas gerada com sucesso.")
print("\n📊 Erro MSE por sensor:")
for col, mse in mse_historico.items():
    print(f"- {col}: {mse:.6f}")
