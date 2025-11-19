FROM python:3.10-slim

RUN pip install --no-cache-dir mlflow==3.5.1

WORKDIR /mlflow

RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1

LABEL maintainer="MLOps Team" \
      version="1.0.0" \
      description="MLflow Tracking Server"

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "*"]
