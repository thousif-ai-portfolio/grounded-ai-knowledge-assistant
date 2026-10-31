"""LLM-as-judge: a language model grades each answer for faithfulness and correctness."""

import json
import re
from dataclasses import dataclass

from app.llm import LLM

# A 3B judge marked correct answers wrong when they were worded differently from the
# reference or added true details from the sources, so the rules and examples below
# spell out that both are fine.
JUDGE_SYSTEM = """You grade answers produced by a question-answering system.

You get the sources the system was given, its answer, and a short reference answer.
Reply with only a JSON object, no other text:
{"faithful": true or false, "correct": true or false}

correct: true when the answer contains the key fact of the reference answer.
- Different wording, extra explanation and extra true details are fine.
- false only when the key fact is missing or contradicted.

faithful: true when the facts in the answer can be found in the sources.
- Paraphrases and summaries of the sources count as supported.
- Citation markers like [1] and phrases like "according to the sources" are fine.
- false only when the answer states a fact that is not in the sources or contradicts them.

Example. Reference: "12 hours." Answer: "Access tokens are valid for 12 hours, so you
sign in twice a day. [1]" Sources say tokens are valid for 12 hours and a new sign-in
is needed twice a day.
{"faithful": true, "correct": true}

Example. Reference: "12 hours." Answer: "Tokens last 24 hours. [1]"
{"faithful": false, "correct": false}"""


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
                faithful=data.get("faithful") is True,
                correct=data.get("correct") is True,
                parsed=True,
            )
        except json.JSONDecodeError:
            pass
    return Verdict(faithful=False, correct=False, parsed=False)
