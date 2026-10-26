# Single-container image for a free Hugging Face Space (Docker SDK, CPU basic).
# Same as the root Dockerfile, plus Ollama and a small answer model baked in,
# because a Space can only run one container. Keep the two files in sync.

FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    OLLAMA_MODELS=/app/.ollama \
    RAG_LLM_MODEL=llama3.2:1b

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates zstd \
 && curl -fsSL https://ollama.com/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app/ app/
COPY evaluation/ evaluation/
COPY scripts/ scripts/
COPY data/sample_docs/ data/sample_docs/
COPY --from=frontend /frontend/dist frontend/dist

RUN python -c "from app.config import get_settings as s; \
from sentence_transformers import CrossEncoder; \
CrossEncoder(s().rerank_model)" \
 && python -m app.cli ingest data/sample_docs \
 && (ollama serve & sleep 5 && ollama pull "$RAG_LLM_MODEL") \
 && chown -R app:app /app

USER app
ENV HF_HUB_OFFLINE=1
# Spaces route traffic to port 7860. Set RAG_JWT_SECRET as a Space secret.
EXPOSE 7860
CMD ["sh", "-c", "ollama serve & exec uvicorn app.main:app --host 0.0.0.0 --port 7860"]
