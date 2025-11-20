import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_wine
import os

def train_model():
    """
    Entrena un modelo de clasificación con scikit-learn.
    Registra métricas y modelo en MLflow.
    """
    # Configurar MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    os.environ["MLFLOW_ARTIFACTS_DESTINATION"] = f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts"
    
    mlflow.set_experiment("wine_classification")
    
    with mlflow.start_run():
        # Cargar datos
        data = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        # Parámetros
        n_estimators = 100
        max_depth = 5
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Entrenar
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluar
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        
        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"\n{'='*50}")
        print(f"Modelo entrenado exitosamente!")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"{'='*50}\n")
        
        # Registrar modelo directamente
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="wine_classification"
        )
        
        return model

if __name__ == "__main__":
    train_model()
