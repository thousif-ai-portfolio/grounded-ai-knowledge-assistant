"""Retrieval-augmented generation: find the right chunks, then answer from them only."""

import hashlib
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

from app.config import Settings
from app.embeddings import Embedder
from app.ingest import chunk_pages, extract_pages
from app.llm import LLM, Generation
from app.retrieval import Bm25Index, Reranker, SearchMode, reciprocal_rank_fusion
from app.vector_store import ChromaStore, DocumentInfo, SearchHit

IDK = "I don't know."

SYSTEM_PROMPT = """You answer questions using only the numbered sources you are given.

Rules:
- Use only facts stated in the sources. Never use outside knowledge.
- Questions may be informal, misspelled or just a keyword such as an error code or a
  setting name. Work out what the user wants to know and answer it if the sources cover it.
- Answer only what was asked. Sources often list several error codes or settings side by
  side; use only the lines about the one in the question.
- End every sentence with the numbers of the sources it came from, like [1] or [2][3].
  Never leave out the citation.
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
    mode: SearchMode
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    sources = "\n\n".join(
        f"[{n}] ({hit.filename}, page {hit.page})\n{hit.text}" for n, hit in enumerate(hits, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def is_idk(text: str) -> bool:
    normalized = text.strip().lower().replace("’", "'")
    return normalized.startswith(("i don't know", "i do not know"))


def cited_numbers(text: str) -> set[int]:
    """Source numbers cited as [1], [2][3] or, as small models sometimes write, [1, 4]."""
    groups = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
    return {int(n) for group in groups for n in group.split(",")}


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        store: ChromaStore,
        llm: LLM,
        settings: Settings,
        reranker: Reranker | None = None,
    ):
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._settings = settings
        self._reranker = reranker
        self._bm25: Bm25Index | None = None

    @property
    def llm(self) -> LLM:
        return self._llm

    def add_document(self, filename: str, data: bytes) -> AddedDocument:
        # The ID comes from the file's bytes, so uploading the same file twice is a no-op.
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        if self._store.has_document(doc_id):
            return AddedDocument(doc_id, filename, chunks=0, already_indexed=True)

        pages = extract_pages(filename, data)
        chunks = chunk_pages(pages, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            raise EmptyDocumentError(f"No text found in '{filename}'. Scanned PDFs need OCR first.")
        embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.add(doc_id, filename, chunks, embeddings)
        self._bm25 = None
        return AddedDocument(doc_id, filename, chunks=len(chunks), already_indexed=False)

    def list_documents(self) -> list[DocumentInfo]:
        return self._store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        deleted = self._store.delete(doc_id)
        if deleted:
            self._bm25 = None
        return deleted

    def _keyword_index(self) -> Bm25Index:
        if self._bm25 is None:
            self._bm25 = Bm25Index(self._store.all_chunks())
        return self._bm25

    def retrieve(
        self, question: str, top_k: int | None = None, mode: SearchMode | None = None
    ) -> list[SearchHit]:
        k = top_k or self._settings.top_k
        mode = SearchMode(mode or self._settings.search_mode)
        candidates = max(k, self._settings.candidates)

        [query_vector] = self._embedder.embed([question])
        semantic = self._store.search(query_vector, candidates)
        # Relevance gate, shared by every mode: if even the closest chunk is far
        # from the question, the documents don't cover it and we answer "I don't know".
        # A borderline score is let through when a rare keyword matches (misspellings).
        floor = self._settings.min_score
        top = semantic[0].score if semantic else 0.0
        if top < floor and self._keyword_index().has_rare_term(question):
            floor = self._settings.keyword_min_score
        if top < floor:
            return []
        if mode is SearchMode.SEMANTIC:
            return [hit for hit in semantic[:k] if hit.score >= floor]

        keyword = self._keyword_index().search(question, candidates)
        by_id = {hit.chunk_id: hit for hit in keyword} | {hit.chunk_id: hit for hit in semantic}
        fused = reciprocal_rank_fusion(
            [[hit.chunk_id for hit in semantic], [hit.chunk_id for hit in keyword]]
        )
        ranked = [replace(by_id[chunk_id], score=score) for chunk_id, score in fused]
        if mode is SearchMode.HYBRID:
            return ranked[:k]

        if self._reranker is None:
            raise ValueError("hybrid_rerank mode needs a reranker")
        pool = ranked[:candidates]
        scores = self._reranker.score(question, [hit.text for hit in pool])
        reranked = sorted(zip(pool, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [replace(hit, score=score) for hit, score in reranked[:k]]

    def ask(
        self, question: str, top_k: int | None = None, mode: SearchMode | None = None
    ) -> Answer:
        *_, answer = self.ask_stream(question, top_k, mode)
        return answer

    def ask_stream(
        self, question: str, top_k: int | None = None, mode: SearchMode | None = None
    ) -> Iterator[str | Answer]:
        """Yields the answer text as the model writes it, then the final Answer, which adds
        the citations and replaces the text with "I don't know." if the model refused."""
        mode = SearchMode(mode or self._settings.search_mode)
        start = time.perf_counter()
        hits = self.retrieve(question, top_k, mode)
        retrieval_ms = _ms_since(start)
        if not hits:
            # Nothing relevant was found, so don't let the model guess.
            yield Answer(IDK, [], grounded=False, mode=mode, retrieval_ms=retrieval_ms)
            return

        start = time.perf_counter()
        parts: list[str] = []
        last = Generation("")
        for last in self._llm.stream(SYSTEM_PROMPT, build_prompt(question, hits)):
            if last.text:
                parts.append(last.text)
                yield last.text
        timing = {
            "mode": mode,
            "retrieval_ms": retrieval_ms,
            "generation_ms": _ms_since(start),
            "input_tokens": last.input_tokens,
            "output_tokens": last.output_tokens,
        }
        text = "".join(parts).strip()
        if not text or is_idk(text):
            yield Answer(IDK, [], grounded=False, **timing)
            return

        cited = cited_numbers(text)
        citations = [
            Citation(number=n, filename=hit.filename, page=hit.page, text=hit.text, score=hit.score)
            for n, hit in enumerate(hits, 1)
            if n in cited
        ]
        yield Answer(text, citations, grounded=bool(citations), **timing)
