FROM python:3.10-slim AS builder

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim AS runtime

# Crear usuario no-root
RUN useradd -m -u 1000 mluser

WORKDIR /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=mluser:mluser . .

# Crear directorios necesarios
RUN mkdir -p /app/models /app/data && \
    chown -R mluser:mluser /app

USER mluser

# Exponer puerto
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Labels para metadatos
LABEL maintainer="MLOps Team" \
      version="1.0.0" \
      description="ML Service - Clasificación Iris"

# Comando de inicio
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
