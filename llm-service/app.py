from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from typing import Optional

app = FastAPI(title="LLM Service API")

class QueryRequest(BaseModel):
    # Accept either 'query' or 'prompt' to be compatible with different clients
    query: Optional[str] = None
    prompt: Optional[str] = None
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

        prompt_text = request.prompt or request.query

        if not prompt_text:
            raise HTTPException(status_code=422, detail="No prompt or query provided")

        payload = {
            "model": os.getenv("LLM_MODEL", "mistral"),
            "prompt": prompt_text,
            "stream": False
        }

        resp = requests.post(f"{ollama_url}/api/generate",
                             json=payload,
                             timeout=120)

        # If Ollama returns an error payload, surface a friendly message
        try:
            resp_json = resp.json()
        except ValueError:
            resp_json = None

        if resp.status_code != 200:
            # If the API returned a structured error, include it; otherwise use status
            if resp_json and isinstance(resp_json, dict):
                error_msg = resp_json.get("error") or resp_json.get("detail") or str(resp_json)
            else:
                error_msg = f"Ollama error: HTTP {resp.status_code}"

            # Provide a helpful fallback instead of raising to the client UI
            # so the chat shows a readable message.
            return QueryResponse(response=f"[Ollama] {error_msg}", model=payload["model"])

        # Success path
        return QueryResponse(response=resp.json().get("response", ""), model=payload["model"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud.
    """
    return {"status": "ok"}