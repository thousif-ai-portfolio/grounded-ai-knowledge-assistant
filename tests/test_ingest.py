import pymupdf
import pytest

from app.ingest import Page, UnsupportedFileError, chunk_pages, extract_pages


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_keeps_page_numbers():
    pages = extract_pages("paper.pdf", make_pdf(["first page", "second page"]))

    assert [p.number for p in pages] == [1, 2]
    assert "second page" in pages[1].text


def test_extract_text_and_markdown():
    assert extract_pages("notes.md", b"# Title\nbody")[0].text == "# Title\nbody"
    assert extract_pages("NOTES.TXT", b"hello")[0].number == 1


def test_extract_rejects_unknown_types():
    with pytest.raises(UnsupportedFileError):
        extract_pages("image.png", b"\x89PNG")


def test_extract_rejects_broken_pdfs():
    with pytest.raises(UnsupportedFileError, match="not a readable PDF"):
        extract_pages("renamed.pdf", b"not really a pdf")


def test_chunks_respect_size_and_split_on_spaces():
    words = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_pages([Page(1, words)], size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.text.split()[0].startswith("word") for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_neighbouring_chunks_overlap():
    words = " ".join(f"w{i}" for i in range(200))
    first, second, *_ = chunk_pages([Page(1, words)], size=100, overlap=30)

    assert second.text.split()[0] in first.text.split()


def test_chunks_never_cross_pages():
    chunks = chunk_pages([Page(1, "alpha " * 50), Page(2, "beta " * 50)], size=120, overlap=20)

    assert {c.page for c in chunks if "alpha" in c.text} == {1}
    assert {c.page for c in chunks if "beta" in c.text} == {2}


def test_blank_pages_produce_no_chunks():
    assert chunk_pages([Page(1, "   \n\t ")], size=100, overlap=10) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], size=50, overlap=50)
