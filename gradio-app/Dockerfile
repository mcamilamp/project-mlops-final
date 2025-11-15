FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Crear usuario no-root
RUN useradd -m -u 1000 gradiouser

WORKDIR /app

# Copiar requirements primero (mejor cache)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY --chown=gradiouser:gradiouser . .

# Cambiar a usuario no-root
USER gradiouser

# Exponer puerto de Gradio
EXPOSE 7860

# Variables de entorno para servicios
ENV LLM_SERVICE_URL=http://llm-service:8000/chat \
    ML_SERVICE_URL=http://ml-service:8000/predict \
    CNN_SERVICE_URL=http://cnn-service:8000/classify

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

# Labels
LABEL maintainer="MLOps Team" \
      version="1.0.0" \
      description="Gradio Frontend - MLOps Project"

# Comando de inicio
CMD ["python", "app.py"]
