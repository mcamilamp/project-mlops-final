""""
ML Service API - Wine Classification Prediction Service

Implementa una API REST usando FastAPI para servir un modelo de clasificación de vinos
entrenado con MLflow. 
Espera 13 características del Wine dataset y devuelve la clase predicha y las probabilidades asociadas. 
También esta la opción de unas características predeterminadas para pruebas rápidas.  

Autor: María Camila Mercado Payares 
Proyecto: MLOps Final

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from typing import List
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Service - Wine Classification")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]

model = None

try:
    logger.info("Intentando cargar modelo wine_classification...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model = mlflow.sklearn.load_model("models:/wine_classification/Production")
    logger.info("Modelo wine_classification versión 1 cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar modelo: {e}")
    logger.info("Entrena y registra un modelo primero con: docker compose exec ml-service python train.py")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Realiza predicción con el modelo entrenado.
    Espera 13 características del Wine dataset.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train and register a model first using train.py"
        )

    try:
        if len(request.features) != 13:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 13 features, got {len(request.features)}"
            )

        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()

        return PredictionResponse(
            prediction=int(prediction),
            probability=probabilities
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
