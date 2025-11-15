import gradio as gr
import requests
from PIL import Image
import io

LLM_URL = "http://llm-service:8000/chat"
ML_URL = "http://ml-service:8000/predict"
CNN_URL = "http://cnn-service:8000/classify"

def chat_with_llm(message, history):
    """
    Interfaz de chat con el LLM.
    """
    try:
        response = requests.post(
            LLM_URL,
            json={"prompt": message},
            timeout=30
        )
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

def classify_with_ml(feature1, feature2, feature3, feature4):
    """
    Clasificación con modelo ML clásico.
    """
    try:
        response = requests.post(
            ML_URL,
            json={"features": [feature1, feature2, feature3, feature4]},
            timeout=10
        )
        result = response.json()
        return f"Clase predicha: {result['prediction']}\nProbabilidades: {result['probability']}"
    except Exception as e:
        return f"Error: {str(e)}"

def classify_image(image):
    """
    Clasificación de imágenes con CNN.
    """
    try:
        # Convertir imagen a bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(CNN_URL, files=files, timeout=30)
        result = response.json()
        
        return f"""
        **Predicción:** {result['prediction']}
        **Confianza:** {result['confidence']:.2%}
        **Clases soportadas:** {', '.join(result['supported_classes'])}
        **Filtros aplicados:** {', '.join(result['filters_applied'])}
        
        {result['warning']}
        """
    except Exception as e:
        return f"Error: {str(e)}"

# Crear interfaz con tabs
with gr.Blocks(title="MLOps Project") as demo:
    gr.Markdown("# Sistema MLOps Integrado")
    gr.Markdown("Integración de LLM, ML Clásico y CNN con buenas prácticas MLOps")
    
    with gr.Tab("Chat LLM"):
        chatbot = gr.ChatInterface(
            fn=chat_with_llm,
            title="Asistente de Lenguaje",
            description="Conversa con el modelo de lenguaje"
        )
    
    with gr.Tab("Clasificación ML"):
        gr.Markdown("### Modelo de clasificación Iris")
        with gr.Row():
            f1 = gr.Number(label="Longitud Sépalo")
            f2 = gr.Number(label="Ancho Sépalo")
            f3 = gr.Number(label="Longitud Pétalo")
            f4 = gr.Number(label="Ancho Pétalo")
        
        ml_button = gr.Button("Clasificar")
        ml_output = gr.Textbox(label="Resultado")
        ml_button.click(classify_with_ml, inputs=[f1, f2, f3, f4], outputs=ml_output)
    
    with gr.Tab("Clasificación de Imágenes"):
        gr.Markdown("### Red Neuronal Convolucional")
        image_input = gr.Image(type="pil", label="Sube una imagen")
        cnn_button = gr.Button("Clasificar Imagen")
        cnn_output = gr.Markdown()
        cnn_button.click(classify_image, inputs=image_input, outputs=cnn_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
