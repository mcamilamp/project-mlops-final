from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from typing import List

app = FastAPI(title="ML Service")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]

# Cargar modelo
model = mlflow.sklearn.load_model("models:/iris_classification/production")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Realiza predicción con el modelo entrenado.
    """
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
