"""
Interfaz en gradio de los tres módulos LLM, ML y CNN
Autor: Maria Camila Mercado Payares
Fecha: 19/11/25
"""


import gradio as gr
import requests
from PIL import Image
import io


LLM_URL = "http://llm-service:8000/chat"
ML_URL = "http://ml-service:8000/predict"  
CNN_URL = "http://cnn-service:8000/classify"


def clean_output(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("[OUT]", "")
            .replace("[/OUT]", "")
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("[B_INST]", "")
            .replace("[/B_INST]", "")
            .strip()
    )


def chat_with_llm(message, history):
    """
    Interfaz de chat con el LLM.
    """
    try:
        response = requests.post(
            LLM_URL,
            json={"prompt": message},
            timeout=120
        )
        logged_response = response.json()
        print(logged_response)
        return clean_output(logged_response['response'])
    except Exception as e:
        print(str(e))
        return f"Error: {str(e)}"


def classify_wine(alcohol, malic_acid, ash, alcalinity, magnesium, 
                  phenols, flavanoids, nonflavanoid, proanthocyanins,
                  color_intensity, hue, od280_od315, proline):
    """
    Clasificación de vinos con modelo ML.
    Wine dataset tiene 13 características.
    """
    try:
        features = [
            alcohol, malic_acid, ash, alcalinity, magnesium,
            phenols, flavanoids, nonflavanoid, proanthocyanins,
            color_intensity, hue, od280_od315, proline
        ]
        
        response = requests.post(
            ML_URL,
            json={"features": features},
            timeout=10
        )
        result = response.json()

        print(result)
        
        wine_classes = {
            0: "Clase 0 - Cultivar 1",
            1: "Clase 1 - Cultivar 2", 
            2: "Clase 2 - Cultivar 3"
        }
        
        prediction_class = wine_classes.get(result['prediction'], "Desconocida")
        probabilities = result['probability']
        
        return f"""
**Predicción:** {prediction_class}

**Probabilidades por clase:**
- Clase 0 (Cultivar 1): {probabilities[0]:.2%}
- Clase 1 (Cultivar 2): {probabilities[1]:.2%}
- Clase 2 (Cultivar 3): {probabilities[2]:.2%}

**Características analizadas:** 13
        """
    except Exception as e:
        return f"Error: {str(e)}"

def classify_image(image):
    """
    Clasificación de imágenes con CNN.
    """
    try:
       
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
        """
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="MLOps Wine Classification", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  Sistema MLOps - Wine Classification")
    gr.Markdown("Integración de LLM, ML Clásico (Wine Dataset) y CNN con buenas prácticas MLOps")
    
    with gr.Tab(" Chat LLM"):
        chatbot = gr.ChatInterface(
            fn=chat_with_llm,
            title="Asistente de Lenguaje",
            description="Conversa con el modelo de lenguaje Ollama"
        )
    
    with gr.Tab(" Clasificación de Vinos"):
        gr.Markdown("""
       
        Clasifica vinos en 3 cultivares basándose en 13 características químicas.
        
        **Características del dataset:**
        - 178 muestras de vinos italianos
        - 3 cultivares diferentes
        - 13 atributos químicos
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Características principales")
                alcohol = gr.Number(label="1. Alcohol (%)", value=13.2)
                malic_acid = gr.Number(label="2. Ácido málico (g/L)", value=2.77)
                ash = gr.Number(label="3. Ceniza (g/L)", value=2.51)
                alcalinity = gr.Number(label="4. Alcalinidad de ceniza", value=18.5)
                magnesium = gr.Number(label="5. Magnesio (mg/L)", value=96.0)
            
            with gr.Column():
                gr.Markdown("#### Características fenólicas")
                phenols = gr.Number(label="6. Fenoles totales", value=1.9)
                flavanoids = gr.Number(label="7. Flavonoides", value=0.58)
                nonflavanoid = gr.Number(label="8. Fenoles no flavonoides", value=0.28)
                proanthocyanins = gr.Number(label="9. Proantocianinas", value=0.42)
            
            with gr.Column():
                gr.Markdown("#### Características de color")
                color_intensity = gr.Number(label="10. Intensidad de color", value=1.95)
                hue = gr.Number(label="11. Matiz", value=1.05)
                od280_od315 = gr.Number(label="12. OD280/OD315", value=1.05)
                proline = gr.Number(label="13. Prolina (mg/L)", value=920.0)
        
        gr.Markdown("#### Ejemplos pre-cargados")
        gr.Examples(
            examples=[
                [13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 0.58, 0.28, 0.42, 1.95, 1.05, 1.05, 920.0], 
                [12.37, 1.13, 2.16, 19.0, 87.0, 3.5, 3.1, 0.19, 1.87, 4.45, 1.22, 2.87, 420.0], 
                [13.4, 3.91, 2.48, 23.0, 102.0, 1.8, 0.75, 0.43, 1.41, 7.3, 0.7, 1.56, 750.0]  
            ],
            inputs=[alcohol, malic_acid, ash, alcalinity, magnesium, phenols, 
                   flavanoids, nonflavanoid, proanthocyanins, color_intensity, 
                   hue, od280_od315, proline],
            label="Haz click en un ejemplo para cargarlo"
        )
        
        ml_button = gr.Button(" Clasificar Vino", variant="primary")
        ml_output = gr.Markdown()
        
        ml_button.click(
            classify_wine, 
            inputs=[alcohol, malic_acid, ash, alcalinity, magnesium, phenols, 
                   flavanoids, nonflavanoid, proanthocyanins, color_intensity, 
                   hue, od280_od315, proline], 
            outputs=ml_output
        )
    
    with gr.Tab("Clasificación de Imágenes"):
        gr.Markdown("### Red Neuronal Convolucional")
        gr.Markdown("Sube una imagen para clasificarla usando el modelo CNN")
        
        image_input = gr.Image(type="pil", label="Sube una imagen")
        cnn_button = gr.Button("Clasificar Imagen", variant="primary")
        cnn_output = gr.Markdown()
        
        cnn_button.click(classify_image, inputs=image_input, outputs=cnn_output)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )
