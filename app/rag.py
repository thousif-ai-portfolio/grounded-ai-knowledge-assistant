"""Retrieval-augmented generation: find the right chunks, then answer from them only."""

import hashlib
import re
from dataclasses import dataclass

from app.config import Settings
from app.embeddings import Embedder
from app.ingest import chunk_pages, extract_pages
from app.llm import LLM
from app.vector_store import ChromaStore, DocumentInfo, SearchHit

IDK = "I don't know."

SYSTEM_PROMPT = """You answer questions using only the numbered sources you are given.

Rules:
- Use only facts stated in the sources. Never use outside knowledge.
- After each sentence, cite the sources it came from, like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly: I don't know."""


class EmptyDocumentError(ValueError):
    """The file contained no text that could be extracted."""


@dataclass(frozen=True)
class AddedDocument:
    doc_id: str
    filename: str
    chunks: int
    already_indexed: bool


@dataclass(frozen=True)
class Citation:
    number: int
    filename: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    # True only when the model cited at least one source it was given.
    grounded: bool


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    sources = "\n\n".join(
        f"[{n}] ({hit.filename}, page {hit.page})\n{hit.text}" for n, hit in enumerate(hits, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def is_idk(text: str) -> bool:
    normalized = text.strip().lower().replace("’", "'")
    return normalized.startswith(("i don't know", "i do not know"))


class RagService:
    def __init__(self, embedder: Embedder, store: ChromaStore, llm: LLM, settings: Settings):
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._settings = settings

    def add_document(self, filename: str, data: bytes) -> AddedDocument:
        # The ID comes from the file's bytes, so uploading the same file twice is a no-op.
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        if self._store.has_document(doc_id):
            return AddedDocument(doc_id, filename, chunks=0, already_indexed=True)

        pages = extract_pages(filename, data)
        chunks = chunk_pages(pages, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            raise EmptyDocumentError(
                f"No text found in '{filename}'. Scanned PDFs need OCR first."
            )
        embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.add(doc_id, filename, chunks, embeddings)
        return AddedDocument(doc_id, filename, chunks=len(chunks), already_indexed=False)

    def list_documents(self) -> list[DocumentInfo]:
        return self._store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        return self._store.delete(doc_id)

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchHit]:
        [query_vector] = self._embedder.embed([question])
        hits = self._store.search(query_vector, top_k or self._settings.top_k)
        return [hit for hit in hits if hit.score >= self._settings.min_score]

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = self.retrieve(question, top_k)
        if not hits:
            # Nothing relevant was found, so don't let the model guess.
            return Answer(IDK, citations=[], grounded=False)

        text = self._llm.generate(SYSTEM_PROMPT, build_prompt(question, hits)).strip()
        if not text or is_idk(text):
            return Answer(IDK, citations=[], grounded=False)

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        citations = [
            Citation(number=n, filename=hit.filename, page=hit.page, text=hit.text, score=hit.score)
            for n, hit in enumerate(hits, 1)
            if n in cited
        ]
        return Answer(text, citations=citations, grounded=bool(citations))
