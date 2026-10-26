"""LLM-as-judge: a language model grades each answer for faithfulness and correctness."""

import json
import re
from dataclasses import dataclass

from app.llm import LLM

JUDGE_SYSTEM = """You grade answers produced by a question-answering system.

You get the sources the system was given, its answer, and a reference answer.
Reply with only a JSON object, no other text:
{"faithful": true or false, "correct": true or false}

- faithful: every factual claim in the answer is supported by the sources.
- correct: the answer agrees with the reference answer (wording may differ)."""


@dataclass(frozen=True)
class Verdict:
    faithful: bool
    correct: bool
    parsed: bool  # False when the judge's reply wasn't valid JSON


def judge_answer(
    llm: LLM, question: str, sources: list[str], answer: str, reference: str
) -> Verdict:
    numbered = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(sources, 1))
    prompt = (
        f"Sources:\n\n{numbered}\n\nQuestion: {question}\n\n"
        f"Answer to grade: {answer}\n\nReference answer: {reference}"
    )
    reply = llm.generate(JUDGE_SYSTEM, prompt).text
    return parse_verdict(reply)


def parse_verdict(reply: str) -> Verdict:
    # Small models sometimes wrap the JSON in prose or code fences; take the first object.
    match = re.search(r"\{.*?\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return Verdict(
                faithful=bool(data.get("faithful")), correct=bool(data.get("correct")), parsed=True
            )
        except json.JSONDecodeError:
            pass
    return Verdict(faithful=False, correct=False, parsed=False)
