"""Turn uploaded files into small, page-tagged text chunks."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int


def extract_pages(filename: str, data: bytes) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                return [Page(number=i + 1, text=page.get_text()) for i, page in enumerate(doc)]
        except pymupdf.FileDataError as exc:
            raise UnsupportedFileError(f"'{filename}' is not a readable PDF") from exc
    if suffix in {".txt", ".md"}:
        return [Page(number=1, text=data.decode("utf-8", errors="replace"))]
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise UnsupportedFileError(f"Unsupported file type '{suffix}'. Supported: {supported}")


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Split each page into chunks of at most `size` characters.

    Chunks never cross a page boundary, so every chunk can be cited by page.
    Neighbouring chunks share about `overlap` characters so a sentence cut at a
    boundary still appears whole in one of them. Splits happen at spaces.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")

    chunks: list[Chunk] = []
    for page in pages:
        text = " ".join(page.text.split())
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page.number, index=len(chunks)))
            if end >= len(text):
                break

            next_start = end - overlap
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
            start = max(next_start, start + 1)
    return chunks
