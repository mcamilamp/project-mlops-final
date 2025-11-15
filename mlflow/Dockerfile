FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 mlflowuser

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow

RUN mkdir -p /mlflow/artifacts /mlflow/data && chown -R mlflowuser:mlflowuser /mlflow

COPY --chown=mlflowuser:mlflowuser . .

USER mlflowuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000')" || exit 1

LABEL maintainer="MLOps Team" version="1.0.0"

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--artifacts-destination", "/mlflow/artifacts", "--backend-store-uri", "sqlite:///mlflow/mlflow.db"]
