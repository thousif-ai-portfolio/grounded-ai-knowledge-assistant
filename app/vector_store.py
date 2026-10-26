"""Store chunk vectors in ChromaDB and find the closest ones to a question."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingest import Chunk

# Chroma rejects very large single inserts, so big PDFs are added in batches.
_ADD_BATCH_SIZE = 1000


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    doc_id: str
    filename: str
    page: int
    score: float  # cosine similarity: 1.0 means identical meaning


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    filename: str
    pages: int
    chunks: int


class ChromaStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = client.get_or_create_collection(collection_name)

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def add(
        self, doc_id: str, filename: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            batch = chunks[start : start + _ADD_BATCH_SIZE]
            self._collection.add(
                ids=[f"{doc_id}:{chunk.index}" for chunk in batch],
                embeddings=embeddings[start : start + _ADD_BATCH_SIZE],
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {"doc_id": doc_id, "filename": filename, "page": chunk.page} for chunk in batch
                ],
            )

    def search(self, embedding: list[float], k: int) -> list[SearchHit]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=str(meta["doc_id"]),
                    filename=str(meta["filename"]),
                    page=int(meta["page"]),
                    # Chroma returns squared L2 distance; for unit vectors
                    # cosine similarity = 1 - distance / 2.
                    score=1 - distance / 2,
                )
            )
        return hits

    def all_chunks(self) -> list[SearchHit]:
        """Every stored chunk (score 0), used to build the keyword index."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            SearchHit(
                chunk_id=chunk_id,
                text=text,
                doc_id=str(meta["doc_id"]),
                filename=str(meta["filename"]),
                page=int(meta["page"]),
                score=0.0,
            )
            for chunk_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]

    def list_documents(self) -> list[DocumentInfo]:
        docs: dict[str, dict] = {}
        for meta in self._collection.get(include=["metadatas"])["metadatas"]:
            doc = docs.setdefault(
                str(meta["doc_id"]), {"filename": str(meta["filename"]), "pages": 0, "chunks": 0}
            )
            doc["pages"] = max(doc["pages"], int(meta["page"]))
            doc["chunks"] += 1
        return [DocumentInfo(doc_id=doc_id, **info) for doc_id, info in docs.items()]

    def delete(self, doc_id: str) -> bool:
        if not self.has_document(doc_id):
            return False
        self._collection.delete(where={"doc_id": doc_id})
        return True
