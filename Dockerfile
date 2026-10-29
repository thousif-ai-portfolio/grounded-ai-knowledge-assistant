# --- 1. Build the React app -------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- 2. Python API that also serves the built React app ----------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app/ app/
COPY evaluation/ evaluation/
COPY data/sample_docs/ data/sample_docs/
COPY --from=frontend /frontend/dist frontend/dist

# Download the embedding and reranker models and index the sample documents at
# build time, so the container starts in seconds and works offline.
RUN python -c "from app.config import get_settings as s; \
from sentence_transformers import CrossEncoder; \
CrossEncoder(s().rerank_model)" \
 && python -m app.cli ingest data/sample_docs \
 && chown -R app:app /app

USER app
# Models are already in the image; don't contact the Hugging Face Hub at runtime.
ENV HF_HUB_OFFLINE=1
EXPOSE 8000
# RAG_JWT_SECRET should be set at runtime; see docker-compose.yml.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
