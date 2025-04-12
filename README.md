# LSTM Predictor API

Uma API para previsão de séries temporais usando LSTM (Long Short-Term Memory) com suporte a GPU via TensorFlow.

## 🚀 Requisitos do Sistema

- Windows 11 ou Windows 10 (versão 21H1 ou superior)
- WSL2 instalado e configurado
- Docker Desktop com suporte WSL2
- NVIDIA GPU compatível com CUDA
- Drivers NVIDIA atualizados (versão 570.x ou superior)
- NVIDIA Container Toolkit instalado

## 📋 Estrutura do Projeto

```
.
├── _Docker/
│   ├── Dockerfile
│   ├── app.py
│   ├── ia_model.py
│   └── requirements.txt
├── Dev/
│    └── *
├── Tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_ia_model.py
│   └── requirements-test.txt
└── README.md
```

## 📊 Visualização do Modelo

![Modelo LSTM](Dev/Modelo%204%20-%20Previsao%20-%20055/previsao_lstm_futuro_multivariavel_055.png)

## ⚙️ Configuração do Ambiente

### 1. Configurar WSL2

```powershell
# Instalar WSL2
wsl --install

# Atualizar WSL2
wsl --update

# Verificar instalação
wsl --status
```

### 2. Instalar NVIDIA Container Toolkit no WSL2

```bash
# Entrar no WSL2
wsl

# Instalar NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Configurar Docker Desktop

1. Abrir `%USERPROFILE%\.docker\daemon.json` e configurar:

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "default-runtime": "nvidia",
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

2. Reiniciar o Docker Desktop

## 🛠️ Instalação

1. Clonar o repositório:
```bash
git clone <repository-url>
cd lstm-predictor
```

2. Construir a imagem Docker:
```bash
docker build -t lstm-predictor .
```

## 🚀 Execução

1. Executar o container com suporte a GPU:
```bash
docker run --gpus all -p 5000:5000 lstm-predictor
```

2. Verificar se a GPU está sendo reconhecida:
```bash
# No host
docker exec -it <container-id> nvidia-smi
```

## 📝 Uso da API

### Endpoint de Previsão

`POST /predict`

Exemplo de payload:
```json
{
    "n_periodos_compar": 20,
    "taxa_de_analise": 0.55,
    "data": [
        {"__time": 1625097600000, "dim": 1, "valor1": 10, "valor2": 20},
        {"__time": 1625184000000, "dim": 1, "valor1": 11, "valor2": 21}
    ]
}
```

## 🧪 Testes

1. Executar testes unitários:
```bash
# No diretório tests/
python -m pytest test_api.py
```

2. Testar GPU no container:
```bash
docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 🔍 Troubleshooting

### Verificar Status da GPU

```bash
# No Windows
nvidia-smi

# No WSL2
wsl nvidia-smi
```

### Logs do Container

```bash
# Ver logs em tempo real
docker logs -f <container-id>
```

### Problemas Comuns

1. GPU não reconhecida:
   - Verificar instalação dos drivers NVIDIA
   - Confirmar configuração do runtime NVIDIA no Docker
   - Verificar se WSL2 está atualizado

2. Erros de memória:
   - Ajustar `batch_size` no modelo
   - Verificar memória disponível na GPU

## 🔧 Variáveis de Ambiente

- `NVIDIA_VISIBLE_DEVICES`: Controla quais GPUs estão visíveis (default: all)
- `CUDA_VISIBLE_DEVICES`: Controla quais GPUs o TensorFlow usa
- `TF_FORCE_GPU_ALLOW_GROWTH`: Alocação dinâmica de memória GPU

## 📚 Referências

- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
- [Docker NVIDIA Runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu)

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes. 