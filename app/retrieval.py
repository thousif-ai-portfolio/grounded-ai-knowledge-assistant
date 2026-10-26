"""Search strategies beyond plain vector search: BM25 keywords, fusion, and reranking."""

import re
from enum import StrEnum
from typing import Protocol

from rank_bm25 import BM25Okapi

from app.vector_store import SearchHit

_TOKEN = re.compile(r"\w+")


class SearchMode(StrEnum):
    SEMANTIC = "semantic"  # vector search only
    HYBRID = "hybrid"  # vector + BM25, merged with Reciprocal Rank Fusion
    HYBRID_RERANK = "hybrid_rerank"  # hybrid, then reordered by a cross-encoder


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Index:
    """Keyword index over every stored chunk. Rebuilt whenever documents change."""

    def __init__(self, chunks: list[SearchHit]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[SearchHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # A zero score means no query word appears in the chunk at all.
        return [self._chunks[i] for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked ID lists: each list adds 1 / (k + rank) to every ID it contains.

    Only ranks matter, not raw scores, so lists with different score scales
    (cosine similarity vs BM25) can be combined fairly.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, 1):
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Higher means more relevant to the query."""
        ...


class CrossEncoderReranker:
    """Reads question and passage together, which is slower but more accurate than vectors."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # loaded on first use so startup stays fast

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return [float(s) for s in self._model.predict([(query, text) for text in texts])]
