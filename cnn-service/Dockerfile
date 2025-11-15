FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Instalar dependencias del sistema necesarias para OpenCV y TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim AS runtime

# Instalar solo las librerías necesarias en runtime
# Para CNN/OpenCV/TensorFlow, podríamos necesitar libgomp1
# pero lo dejaremos mínimo sin librerías gráficas obsoletas

# Crear usuario no-root
RUN useradd -m -u 1000 cnnuser

WORKDIR /app

# Copiar dependencias de Python desde builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar código fuente
COPY --chown=cnnuser:cnnuser . .

# Crear directorios para modelos y datos
RUN mkdir -p /app/models /app/data && \
    chown -R cnnuser:cnnuser /app

# Cambiar a usuario no-root
USER cnnuser

# Exponer puerto
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Labels
LABEL maintainer="MLOps Team" \
      version="1.0.0" \
      description="CNN Service - Clasificación de Imágenes"

# Comando de inicio
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
