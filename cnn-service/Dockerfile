FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install httpx pytest

COPY . .

EXPOSE 8000

CMD pytest test.py && uvicorn predict:app --host 0.0.0.0 --port 8000 --workers 1
