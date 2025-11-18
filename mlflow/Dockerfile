FROM mlflow:latest

WORKDIR /mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000')" || exit 1

LABEL maintainer="MLOps Team" version="1.0.0"

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "*"]
