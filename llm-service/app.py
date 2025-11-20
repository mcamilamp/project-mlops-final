
"""
LLM Service API - Servicio de Consulta a Modelos de Lenguaje

Se implementa una API REST usando FAST API que permite a los usuarios hacer consultas
a modelos de lenguaje. Primero intenta usar un modelo local a través de Ollama, si Ollama falla
entonces usa OpenRouter como respaldo.

Autor: María Camila Mercado Payares 
Proyecto: MLOps Final
"""


import os
import requests
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Service API")


class QueryRequest(BaseModel):
    query: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    model: str


def call_ollama(prompt_text: str, model_name: str) -> Optional[str]:
    
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    logger.info(f"[OLLAMA] Intentando consulta al modelo '{model_name}' en {ollama_url}")

    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False
    }

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=10
        )

        logger.info(f"[OLLAMA] Código de estado recibido: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            logger.info("[OLLAMA] Respuesta exitosa recibida.")
            return data.get("response", "")

        logger.warning(f"[OLLAMA] Error HTTP {resp.status_code}. Activando fallback.")
        return None

    except Exception as e:
        logger.error(f"[OLLAMA] Falló la conexión o procesamiento: {e}")
        return None


def call_openrouter(prompt_text: str, model_name: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.critical("[OPENROUTER] Falta OPENROUTER_API_KEY en variables de entorno.")
        raise HTTPException(500, "Missing OPENROUTER_API_KEY")

    logger.info(f"[OPENROUTER] Llamando al modelo remoto '{model_name}'")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
        )

        logger.info("[OPENROUTER] Respuesta recibida correctamente.")
        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"[OPENROUTER] Error al procesar la solicitud: {e}")
        raise HTTPException(status_code=500, detail=f"OpenRouter error: {e}")



@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    prompt_text = request.prompt or request.query
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Debes enviar 'query' o 'prompt'.")


    ollama_model = os.getenv("LLM_MODEL", "orca-mini")
    open_router_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.1")

    logger.info("[CHAT] Intentando primero con Ollama...")
    ollama_response = call_ollama(prompt_text, ollama_model)

    if ollama_response:
        logger.info("[CHAT] Respuesta obtenida desde Ollama.")
        return QueryResponse(response=ollama_response, model=f"Ollama:{ollama_model}")

    logger.warning("[CHAT] Ollama no disponible. Usando OpenRouter como fallback.")
    remote_response = call_openrouter(prompt_text, open_router_model)

    return QueryResponse(response=remote_response, model=f"OpenRouter:{open_router_model}")

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }
