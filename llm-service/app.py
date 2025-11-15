from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from typing import Optional

app = FastAPI(title="LLM Service API")

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    model: str

@app.post("/chat", response_model=QueryResponse)

async def chat(request: QueryRequest):
    """
    Endpoint para consultar un modelo de lenguaje (LLM).
    """
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        payload = {
            "model": os.getenv("LLM_MODEL", "llama2"),
            "prompt": request.prompt,
            "stream": False
        }

        response = requests.post(f"{ollama_url}/api/generate",
        json=payload,
        timeout=30
        )
        response.raise_for_status()

        return QueryResponse(
            response=response.json()["response"],
            model=payload["model"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud.
    """
    return {"status": "ok"}