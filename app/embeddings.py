"""Turn text into vectors so similar meanings end up close together."""

from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-length vector per text."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        # Imported here so tests that use a fake embedder don't load PyTorch.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()
