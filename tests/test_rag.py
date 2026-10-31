import pytest

from app.rag import IDK, EmptyDocumentError, cited_numbers, is_idk

DOC = b"""Chunk overlap repeats a few hundred characters between neighbouring chunks.
BM25 is a keyword ranking function that rewards rare query words."""


def test_answer_cites_the_sources_the_model_used(rag, llm):
    rag.add_document("notes.md", DOC)

    answer = rag.ask("What does chunk overlap repeat?")

    assert answer.grounded
    assert [c.number for c in answer.citations] == [1]
    assert answer.citations[0].filename == "notes.md"
    assert answer.citations[0].page == 1
    assert "[1] (notes.md, page 1)" in llm.prompts[0]


def test_uploading_the_same_file_twice_is_a_no_op(rag):
    first = rag.add_document("notes.md", DOC)
    second = rag.add_document("copy.md", DOC)

    assert not first.already_indexed and first.chunks > 0
    assert second.already_indexed and second.doc_id == first.doc_id
    assert len(rag.list_documents()) == 1


def test_no_documents_means_idk_without_calling_the_model(rag, llm):
    answer = rag.ask("What is BM25?")

    assert answer.answer == IDK and not answer.grounded
    assert llm.prompts == []


def test_unrelated_question_is_filtered_by_min_score(rag, llm):
    rag.add_document("notes.md", DOC)

    answer = rag.ask("zebra giraffe safari")

    assert answer.answer == IDK
    assert llm.prompts == []


def test_model_refusal_returns_idk(rag, llm):
    rag.add_document("notes.md", DOC)
    llm.reply = "I don’t know, the sources do not say."

    answer = rag.ask("What does chunk overlap repeat?")

    assert answer.answer == IDK and answer.citations == []


def test_answer_without_citations_is_not_grounded(rag, llm):
    rag.add_document("notes.md", DOC)
    llm.reply = "Overlap repeats text."

    answer = rag.ask("What does chunk overlap repeat?")

    assert not answer.grounded and answer.citations == []


def test_citations_to_unknown_sources_are_ignored(rag, llm):
    rag.add_document("notes.md", DOC)
    llm.reply = "Made up [99]."

    assert rag.ask("What does chunk overlap repeat?").citations == []


def test_empty_document_is_rejected(rag):
    with pytest.raises(EmptyDocumentError):
        rag.add_document("blank.txt", b"   ")


def test_delete_document(rag):
    added = rag.add_document("notes.md", DOC)

    assert rag.delete_document(added.doc_id)
    assert not rag.delete_document(added.doc_id)
    assert rag.list_documents() == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I don't know.", True),
        ("i do not know", True),
        ("I don't know. [1]", True),
        ("I don't know; the sources don't say.", True),
        ("I don't know the exact date, but [1] says 2020.", False),
        ("I know this [1]", False),
    ],
)
def test_is_idk(text, expected):
    assert is_idk(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Fact [1]. Other [2][3].", {1, 2, 3}),
        ("Error E1042: the cache is full. [1, 4]", {1, 4}),
        ("No citation, just a list [a, b] and E1042.", set()),
    ],
)
def test_cited_numbers(text, expected):
    assert cited_numbers(text) == expected


def test_comma_separated_citations_are_kept(rag, llm):
    rag.add_document("notes.md", DOC)
    llm.reply = "Overlap repeats text [1, 2]."

    answer = rag.ask("What does chunk overlap repeat?")

    assert answer.grounded and [c.number for c in answer.citations] == [1]
