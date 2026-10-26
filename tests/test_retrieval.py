import pytest

from app.retrieval import Bm25Index, SearchMode, reciprocal_rank_fusion, tokenize
from app.vector_store import SearchHit


def hit(chunk_id: str, text: str) -> SearchHit:
    return SearchHit(chunk_id, text, doc_id="d", filename="f.md", page=1, score=0.0)


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Error E1042: Disk-full!") == ["error", "e1042", "disk", "full"]


def test_bm25_ranks_exact_term_matches_first():
    index = Bm25Index(
        [
            hit("a", "general advice about storage and disks"),
            hit("b", "error E1042 means the cache directory is full"),
            hit("c", "unrelated text about cooking pasta"),
        ]
    )

    results = index.search("what does E1042 mean", k=3)

    assert results[0].chunk_id == "b"
    assert "c" not in [r.chunk_id for r in results]


def test_bm25_on_empty_corpus_returns_nothing():
    assert Bm25Index([]).search("anything", k=5) == []


def test_rrf_rewards_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)

    assert fused[0][0] == "b"
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert {item for item, _ in fused} == {"a", "b", "c", "d"}


DOCS = {
    "overlap.md": b"Chunk overlap repeats characters between neighbouring chunks.",
    "codes.md": b"Error E1042 means the chunk cache is full. Clear the chunk cache.",
    "bm25.md": b"BM25 is a keyword ranking function for chunk search.",
}


@pytest.fixture
def loaded(rag):
    for name, data in DOCS.items():
        rag.add_document(name, data)
    return rag


@pytest.mark.parametrize("mode", list(SearchMode))
def test_every_mode_finds_the_relevant_chunk(loaded, mode):
    hits = loaded.retrieve("What does error E1042 mean?", top_k=3, mode=mode)

    assert "codes.md" in [h.filename for h in hits]


def test_reranker_decides_final_order(loaded, reranker):
    reranker.favourite = "bm25"

    hits = loaded.retrieve("chunk", top_k=3, mode=SearchMode.HYBRID_RERANK)

    assert hits[0].filename == "bm25.md"
    assert reranker.calls == 1


def test_semantic_and_hybrid_modes_skip_the_reranker(loaded, reranker):
    loaded.retrieve("chunk", mode=SearchMode.SEMANTIC)
    loaded.retrieve("chunk", mode=SearchMode.HYBRID)

    assert reranker.calls == 0


def test_off_topic_question_returns_nothing_in_every_mode(loaded):
    for mode in SearchMode:
        assert loaded.retrieve("zebra giraffe safari", mode=mode) == []


def test_keyword_index_updates_after_add_and_delete(loaded):
    added = loaded.add_document("new.md", b"Quokka habitats chunk notes.")
    assert "new.md" in [h.filename for h in loaded.retrieve("quokka chunk", mode="hybrid")]

    loaded.delete_document(added.doc_id)
    assert "new.md" not in [h.filename for h in loaded.retrieve("quokka chunk", mode="hybrid")]
