# Usar una imagen ligera de Python 3.11
FROM python:3.13-slim

# Instalar Git (necesario para la orquestación GitOps)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias y el código fuente
COPY requirements.txt .
COPY src/ ./src/
COPY models/ ./models/
COPY .env .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Configurar Git dentro del contenedor para que no de errores al hacer commits
RUN git config --global user.name "MLOps Bot" && \
    git config --global user.email "bot@mlops.local"

# El comando que mantendrá vivo el contenedor
CMD ["python", "src/folder_watcher.py"]