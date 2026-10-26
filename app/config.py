from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """App settings. Override any of them with RAG_* environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="RAG_", extra="ignore"
    )

    # Anchored to the project folder so the index is the same wherever the app is started.
    data_dir: Path = PROJECT_ROOT / "data" / "index"
    collection_name: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5

    # semantic | hybrid | hybrid_rerank (see app/retrieval.py)
    search_mode: str = "hybrid_rerank"
    # How many results each search method contributes before fusion and reranking.
    candidates: int = 20
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Chunks less similar than this (cosine, 0-1) are ignored, so off-topic
    # questions get "I don't know" instead of a guess.
    min_score: float = 0.25
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
