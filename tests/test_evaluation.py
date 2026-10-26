from collections import Counter

import pytest

from app.config import PROJECT_ROOT
from app.ingest import extract_pages
from app.vector_store import SearchHit
from evaluation.dataset import EvalItem, first_relevant_rank, load_questions, normalize
from evaluation.judge import parse_verdict
from evaluation.metrics import mrr_at, percentile, rate, recall_at

DOCS = PROJECT_ROOT / "data" / "sample_docs"


def test_question_ids_are_unique():
    ids = [item.id for item in load_questions()]
    assert len(ids) == len(set(ids))


def test_dataset_size_and_mix():
    counts = Counter(item.type for item in load_questions())
    assert sum(counts.values()) >= 50
    assert counts["unanswerable"] >= 10
    assert counts["keyword"] >= 10


@pytest.mark.parametrize("item", [i for i in load_questions() if i.answerable], ids=lambda i: i.id)
def test_every_evidence_phrase_exists_in_its_source(item: EvalItem):
    """Guards the labels: a typo in an evidence phrase would silently lower recall."""
    path = DOCS / item.source
    text = " ".join(page.text for page in extract_pages(path.name, path.read_bytes()))
    assert normalize(item.evidence) in normalize(text)


def hit(filename: str, text: str) -> SearchHit:
    return SearchHit("id", text, "doc", filename, 1, 0.5)


def test_first_relevant_rank_needs_matching_file_and_phrase():
    item = EvalItem("x", "q?", "keyword", source="a.md", evidence="Error  E1042")
    hits = [hit("b.md", "error e1042"), hit("a.md", "nothing here"), hit("a.md", "Error\nE1042 x")]

    assert first_relevant_rank(item, hits) == 3
    assert first_relevant_rank(item, hits[:2]) is None


def test_recall_and_mrr():
    ranks = [1, 3, None, 6]

    assert recall_at(ranks, 1) == 0.25
    assert recall_at(ranks, 5) == 0.5
    assert mrr_at(ranks, 5) == pytest.approx((1 + 1 / 3) / 4)


def test_percentile_and_rate():
    values = [float(v) for v in range(1, 101)]

    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([], 95) == 0
    assert rate([True, False, True, True]) == 0.75


@pytest.mark.parametrize(
    ("reply", "faithful", "correct", "parsed"),
    [
        ('{"faithful": true, "correct": false}', True, False, True),
        ('Sure!\n```json\n{"faithful": true, "correct": true}\n```', True, True, True),
        ("I think it is fine.", False, False, False),
    ],
)
def test_parse_verdict(reply, faithful, correct, parsed):
    verdict = parse_verdict(reply)

    assert (verdict.faithful, verdict.correct, verdict.parsed) == (faithful, correct, parsed)
