"""The evaluation question set and the rule for deciding whether a chunk is relevant."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.vector_store import SearchHit

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.jsonl"


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    type: str  # "natural", "keyword" (exact codes, names, numbers) or "unanswerable"
    source: str | None = None  # file that contains the answer
    evidence: str | None = None  # exact phrase from that file that proves the answer
    reference: str | None = None  # short correct answer, used by the answer judge

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_questions(path: Path = QUESTIONS_FILE) -> list[EvalItem]:
    with path.open(encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_relevant(item: EvalItem, filename: str, text: str) -> bool:
    """A chunk is relevant if it comes from the right file and contains the evidence.

    Matching on an evidence phrase instead of chunk IDs keeps the labels valid
    when chunk size or overlap settings change.
    """
    if item.source is None or item.evidence is None:
        return False
    return filename == item.source and normalize(item.evidence) in normalize(text)


def first_relevant_rank(item: EvalItem, hits: list[SearchHit]) -> int | None:
    for rank, hit in enumerate(hits, 1):
        if is_relevant(item, hit.filename, hit.text):
            return rank
    return None
