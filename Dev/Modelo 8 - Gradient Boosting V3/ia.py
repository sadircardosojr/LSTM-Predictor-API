#%%
#%%
import pandas as pd
import numpy as np
import json
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import plotly.graph_objects as go

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
                print(f"Coluna de tempo detectada: {col} (formato: millisegundos)")
                return df, col
            except:
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Coluna de tempo detectada: {col} (formato: data/hora)")
                    return df, col
                except:
                    continue
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].min() > 1e12:
                try:
                    df[col] = pd.to_datetime(df[col], unit='ms')
                    return df, col
                except:
                    continue
            elif df[col].min() > 1e9:
                try:
                    df[col] = pd.to_datetime(df[col], unit='s')
                    return df, col
                except:
                    continue
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                return df, col
            except:
                continue
    raise ValueError("Nenhuma coluna de tempo válida encontrada.")

# === 1. Carregar dados ===
with open(arquivo_json, "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# === 2. Processar tempo ===
try:
    df, time_column = detect_time_column(df)
    df.set_index(time_column, inplace=True)
except ValueError as e:
    print(f"⚠️ {e}")
    df.index = pd.date_range(start='now', periods=len(df), freq='H')
    print("Criado índice temporal artificial.")

if not df.index.is_monotonic_increasing:
    df.sort_index(inplace=True)

if df.index.duplicated().any():
    df = df[~df.index.duplicated(keep='last')]

value_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and df[col].notna().sum() > 0.9 * len(df)]
if not value_cols:
    raise ValueError("Nenhuma coluna numérica válida encontrada.")

print(f"Sensores usados: {value_cols}")
df = df[value_cols].dropna()

# === 3. Criar features derivadas ===
for col in value_cols:
    df[f"{col}_diff1"] = df[col].diff()
    df[f"{col}_ma3"] = df[col].rolling(3).mean()
    df[f"{col}_ma5"] = df[col].rolling(5).mean()

df.dropna(inplace=True)
feature_cols = df.columns.tolist()

# === 4. Janela
total_periods = len(df)
window_size = max(24, min(int(total_periods * TAXA_DE_ANALISE), 1000))
granularidade = df.index.to_series().diff().median()
print(f"Janela: {window_size}, Granularidade: {granularidade}")

# === 5. Normalizar
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size].flatten())
        y.append(data[i+window_size][:len(value_cols)])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, window_size)
X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(X)*0.8)]
X_test, y_test = X[int(len(X)*0.8):], y[int(len(X)*0.8):]

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
    np.hstack([previsoes_historico_scaled, np.zeros((len(previsoes_historico_scaled), len(feature_cols) - len(value_cols)))])
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
    np.hstack([futuro_escalado, np.zeros((n_periodos, len(feature_cols) - len(value_cols)))])
)[:, :len(value_cols)]

datas_futuras = [df.index[-1] + granularidade * (i + 1) for i in range(n_periodos)]

# === 9. Gráfico interativo com Plotly
if PLOT_SOMENTE_ULTIMOS_PERIODOS:
    tempo_minimo = df.index[-1] - PLOT_SOMENTE_ULTIMOS_PERIODOS * granularidade
    df_plot = df[df.index >= tempo_minimo]
else:
    df_plot = df

fig = go.Figure()

for i, col in enumerate(value_cols):
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot[col],
        mode='lines',
        name=f'Histórico Real ({col})',
        line=dict(width=2),
        hovertemplate=f"{col}<br>Data: %{{x|%Y-%m-%d %H:%M}}<br>Valor: %{{y:.2f}}"

    ))

    fig.add_trace(go.Scatter(
        x=datas_futuras,
        y=futuro[:, i],
        mode='lines+markers',
        name=f'Projeção ({col})',
        line=dict(dash='dash', width=2),
        marker=dict(size=6),
        hovertemplate=f"{col} (Proj)<br>Data: %{{x|%Y-%m-%d %H:%M}}<br>Valor: %{{y:.2f}}"

    ))

fig.update_layout(
    title='Histórico + Projeção Gradient Boosting com Features Derivadas',
    xaxis_title='Tempo',
    yaxis_title='Valor',
    hovermode='x unified',
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

data_hora_atual = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_html = f"grafico_interativo_{data_hora_atual}.html"
fig.write_html(nome_html)

print(f"📁 Gráfico interativo salvo como {nome_html} — abra no navegador para visualizar.")

# === 10. Erro
mse_historico = {}
for i, col in enumerate(value_cols):
    mse = mean_squared_error(df[col].values[window_size:], previsoes_historico[:, i])
    mse_historico[col] = mse

print("✅ Análise finalizada com gráfico interativo.")
print("\n📊 Erro MSE por sensor:")
for col, mse in mse_historico.items():
    print(f"- {col}: {mse:.6f}")

