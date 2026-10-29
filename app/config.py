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

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    # Keep the model in memory between questions (Ollama's default is 5m) to skip reloads.
    llm_keep_alive: str = "30m"
    # Answers average ~25 tokens; the cap stops a runaway answer taking minutes on a CPU.
    max_answer_tokens: int = 300

    chunk_size: int = 800
    chunk_overlap: int = 150
    # Reading the chunks is most of the answer time on a CPU. The evaluation sets its own k.
    top_k: int = 3

    # semantic | hybrid | hybrid_rerank (see app/retrieval.py)
    search_mode: str = "hybrid_rerank"
    # How many results each search method contributes before fusion and reranking.
    candidates: int = 20
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # If even the closest chunk is less similar than this (cosine, 0-1), the question
    # is off-topic and gets "I don't know" without calling the model. Chosen from the
    # evaluation set: off-topic questions score 0.07-0.15, the weakest answerable one
    # 0.24. Near-miss questions score higher and are left to the model to refuse.
    min_score: float = 0.2
    # Misspelled questions can score below min_score ("chunck overlap": 0.14). Between
    # this and min_score a question still passes if it shares a rare word with the
    # documents; off-topic questions in that range (0.14-0.15) share none.
    keyword_min_score: float = 0.1
    max_upload_mb: int = 20

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}"
    # Set RAG_JWT_SECRET in production. If empty, a random secret is generated at
    # startup, so everyone has to log in again after a restart.
    jwt_secret: str = ""
    jwt_expire_minutes: int = 12 * 60

    eval_results_dir: Path = PROJECT_ROOT / "evaluation" / "results"
    # The built React app. Served at / when it exists (after `npm run build`).
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    # Used only to report cost in the evaluation. 0 for local models; set these
    # to a hosted model's price per million tokens to compare.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
