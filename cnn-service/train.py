import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mlflow
import mlflow.tensorflow
import os

def create_cnn_model(input_shape=(64, 64, 3), num_classes=3):
    """
    Crea una CNN con tres capas de convolución.
    """
    model = Sequential([
        # Primera capa convolucional - Filtro de detección de bordes
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Segunda capa - Filtro de suavizado
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Tercera capa - Filtro de nitidez
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_cnn():
    """
    Entrena la CNN y registra en MLflow.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("image_classification_cnn")
    
    with mlflow.start_run():
        model = create_cnn_model()
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Log parámetros
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("num_conv_layers", 3)
        mlflow.log_param("num_classes", 3)
        
        # Entrenar (usar datos reales aquí)
        # history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
        
        # Log modelo
        mlflow.tensorflow.log_model(model, "model")
        
        return model

if __name__ == "__main__":
    train_cnn()
