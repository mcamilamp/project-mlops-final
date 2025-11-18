FROM python:3.10-slim

# Instalar curl y cliente PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLflow con soporte PostgreSQL
RUN pip install --no-cache-dir mlflow==2.9.2 psycopg2-binary

WORKDIR /mlflow

# Crear directorio para artifacts
RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1

LABEL maintainer="MLOps Team" \
      version="1.0.0" \
      description="MLflow Tracking Server"

# Comando de inicio usando variables de entorno
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000
