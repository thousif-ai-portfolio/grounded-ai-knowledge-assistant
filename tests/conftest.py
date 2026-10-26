"""Fakes so tests run fast and offline: no model downloads, no Ollama."""

import hashlib
import math
import re

import pytest

from app.config import Settings
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
    def __init__(self, reply: str | Exception = "Chunk overlap repeats text [1].") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "index", chunk_size=200, chunk_overlap=40, min_score=0.1)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def rag(settings, llm) -> RagService:
    store = ChromaStore(settings.data_dir, settings.collection_name)
    return RagService(embedder=FakeEmbedder(), store=store, llm=llm, settings=settings)
