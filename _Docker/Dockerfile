# Imagem base com suporte a GPU, CUDA 12.2, cuDNN 8.9, TF 2.15
FROM tensorflow/tensorflow:2.15.0-gpu

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copiar código da aplicação
COPY app.py .
COPY ia_model.py .

# Expor a porta da API Flask
EXPOSE 5000

# Copiar requisitos e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Comando para iniciar a aplicação
CMD ["python", "app.py", "debug=True"] 