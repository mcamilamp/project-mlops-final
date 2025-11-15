from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2

app = FastAPI(title="CNN Service")

CLASSES = ["perro", "gato", "pájaro"]
SUPPORTED_CLASSES_MSG = f"Este modelo solo reconoce: {', '.join(CLASSES)}"

def apply_filters(image_array):
    """
    Aplica tres filtros de convolución a la imagen.
    """
    # Filtro 1: Suavizado (Gaussian Blur)
    smoothed = cv2.GaussianBlur(image_array, (5, 5), 0)
    
    # Filtro 2: Detección de bordes (Sobel)
    edges = cv2.Sobel(image_array, cv2.CV_64F, 1, 1, ksize=5)
    
    # Filtro 3: Nitidez (Sharpening)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image_array, -1, kernel)
    
    return {
        "smoothed": smoothed,
        "edges": edges,
        "sharpened": sharpened
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Clasifica una imagen y aplica filtros de convolución.
    """
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((64, 64))
        image_array = np.array(image)
        
        # Aplicar filtros
        filtered_images = apply_filters(image_array)
        
        # Normalizar para predicción
        image_normalized = image_array / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Cargar modelo y predecir
        model = tf.keras.models.load_model("model")
        predictions = model.predict(image_batch)
        predicted_class = CLASSES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return JSONResponse({
            "prediction": predicted_class,
            "confidence": confidence,
            "supported_classes": CLASSES,
            "warning": SUPPORTED_CLASSES_MSG,
            "filters_applied": ["suavizado", "bordes", "nitidez"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
