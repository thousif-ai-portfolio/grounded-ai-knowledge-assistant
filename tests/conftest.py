"""Fakes so tests run fast and offline: no model downloads, no Ollama."""

import hashlib
import math
import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import Generation
from app.main import create_app
from app.rag import RagService
from app.vector_store import ChromaStore


class FakeEmbedder:
    """Bag-of-words vectors: texts that share words end up close together."""

    dim = 4096  # large enough that unrelated words rarely collide

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append([x / norm for x in vector])
        return vectors


class FakeLLM:
    model = "fake-model"

    def __init__(self, reply: str | Exception = "Chunk overlap repeats text [1].") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str) -> Generation:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return Generation(self.reply, input_tokens=len(prompt.split()), output_tokens=5)


class FakeReranker:
    """Scores a passage by how many times it contains `favourite`, so tests control order."""

    def __init__(self, favourite: str = "") -> None:
        self.favourite = favourite
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return [
            float(text.lower().count(self.favourite)) if self.favourite else 0.0 for text in texts
        ]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "index",
        database_url=f"sqlite:///{(tmp_path / 'app.db').as_posix()}",
        eval_results_dir=tmp_path / "results",
        frontend_dist=tmp_path / "dist",
        jwt_secret="test-secret-that-is-at-least-32-bytes-long",
        chunk_size=200,
        chunk_overlap=40,
        min_score=0.1,
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def rag(settings, llm, reranker) -> RagService:
    store = ChromaStore(settings.data_dir, settings.collection_name)
    return RagService(
        embedder=FakeEmbedder(), store=store, llm=llm, settings=settings, reranker=reranker
    )


@pytest.fixture
def client(rag, settings) -> TestClient:
    with TestClient(create_app(service=rag, settings=settings)) as client:
        yield client
