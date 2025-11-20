"""
Servidor del modelo cnn clasificador de emociones segun el gesto de la cara.
las emociones soportadas son 'angry', 'disgusted', 'fearful', 'neutral', 'surprised'
el modelo está hecho con tensorflow/keras. en el archivo pipeline/pipeline.ipynb

LIMITACIÖN. El entrenamiento del CNN no se realiza automáticamente.
al ser un archivo notebook dificulta un poco la automatización del entrenamiento.
Además toma cierto tiempo (>10min)

Autor: Miguel Amézquita
Fecha: 21/11/25
"""


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os

app = FastAPI(title="CNN Service")

CLASSES = emotions = ['angry', 'disgusted', 'fearful', 'neutral', 'surprised']
SUPPORTED_CLASSES_MSG = f"Este modelo solo reconoce: {', '.join(CLASSES)}"


@app.get("/supported_classes")
async def get_supported_classes():
    return {"supported_classes": CLASSES, "warning": SUPPORTED_CLASSES_MSG}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.tensorflow.load_model("models:/emotion-classifier/Production")

    """
    Clasifica una imagen
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        image = image.resize((48, 48))
        image_array = np.array(image)
        image_normalized = image_array / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        predictions = model.predict(image_batch)
        predicted_class = CLASSES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return JSONResponse({
            "prediction": predicted_class,
            "confidence": confidence,
            "supported_classes": CLASSES,
            "warning": SUPPORTED_CLASSES_MSG
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
