"""Run the evaluation and save results for the dashboard.

    python -m evaluation.run retrieval            # search quality only, no LLM needed
    python -m evaluation.run answers              # full answers, needs Ollama running
    python -m evaluation.run retrieval --min-recall 0.8   # fail (exit 1) below a threshold

The documents in data/sample_docs are indexed into a temporary folder, so your own
uploaded documents are never touched.
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import PROJECT_ROOT, Settings, get_settings
from app.embeddings import SentenceTransformerEmbedder
from app.ingest import SUPPORTED_EXTENSIONS
from app.llm import LLMError, create_llm
from app.rag import IDK, RagService
from app.retrieval import CrossEncoderReranker, SearchMode
from app.vector_store import ChromaStore
from evaluation.dataset import EvalItem, first_relevant_rank, is_relevant, load_questions
from evaluation.judge import judge_answer
from evaluation.metrics import mrr_at, percentile, rate, recall_at

DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
K = 5


def build_eval_service(index_dir: Path) -> tuple[RagService, Settings]:
    settings = get_settings().model_copy(update={"data_dir": index_dir, "top_k": K})
    rag = RagService(
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        store=ChromaStore(index_dir, "eval"),
        llm=create_llm(settings),
        settings=settings,
        reranker=CrossEncoderReranker(settings.rerank_model),
    )
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            rag.add_document(path.name, path.read_bytes())
    return rag, settings


def config_summary(settings: Settings) -> dict:
    keys = [
        "embedding_model",
        "rerank_model",
        "llm_model",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "candidates",
        "min_score",
    ]
    return {key: getattr(settings, key) for key in keys}


def evaluate_retrieval(rag: RagService, items: list[EvalItem]) -> dict:
    answerable = [item for item in items if item.answerable]
    unanswerable = [item for item in items if not item.answerable]
    modes: dict[str, dict] = {}
    per_question: dict[str, dict] = {item.id: {"type": item.type} for item in items}

    for mode in SearchMode:
        ranks: dict[str, int | None] = {}
        rejected: list[bool] = []
        latencies: list[float] = []
        for item in items:
            start = time.perf_counter()
            hits = rag.retrieve(item.question, top_k=K, mode=mode)
            latencies.append((time.perf_counter() - start) * 1000)
            if item.answerable:
                ranks[item.id] = first_relevant_rank(item, hits)
                per_question[item.id][mode.value] = ranks[item.id]
            else:
                rejected.append(not hits)
                per_question[item.id][mode.value] = "rejected" if not hits else "retrieved"

        by_type = {}
        for kind in sorted({item.type for item in answerable}):
            kind_ranks = [ranks[item.id] for item in answerable if item.type == kind]
            by_type[kind] = {
                "questions": len(kind_ranks),
                "recall_at_5": recall_at(kind_ranks, K),
                "mrr_at_5": mrr_at(kind_ranks, K),
            }
        all_ranks = list(ranks.values())
        modes[mode.value] = {
            "recall_at_1": recall_at(all_ranks, 1),
            "recall_at_5": recall_at(all_ranks, K),
            "mrr_at_5": mrr_at(all_ranks, K),
            "unanswerable_rejected": rate(rejected),
            "latency_p50_ms": round(percentile(latencies, 50), 1),
            "latency_p95_ms": round(percentile(latencies, 95), 1),
            "by_type": by_type,
            "missed": [item_id for item_id, rank in ranks.items() if rank is None or rank > K],
        }

    return {
        "questions": len(items),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "modes": modes,
        "per_question": per_question,
    }


def evaluate_answers(
    rag: RagService, settings: Settings, items: list[EvalItem], mode: SearchMode
) -> dict:
    rows = []
    for n, item in enumerate(items, 1):
        print(f"  [{n}/{len(items)}] {item.id}", flush=True)
        start = time.perf_counter()
        answer = rag.ask(item.question, top_k=K, mode=mode)
        latency = (time.perf_counter() - start) * 1000
        refused = answer.answer == IDK
        row = {
            "id": item.id,
            "type": item.type,
            "question": item.question,
            "answer": answer.answer,
            "refused": refused,
            "latency_ms": round(latency, 1),
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
        }
        if item.answerable and not refused:
            row["citation_hit"] = any(
                is_relevant(item, c.filename, c.text) for c in answer.citations
            )
            sources = [h.text for h in rag.retrieve(item.question, top_k=K, mode=mode)]
            verdict = judge_answer(
                rag.llm, item.question, sources, answer.answer, item.reference or ""
            )
            row |= {
                "faithful": verdict.faithful,
                "correct": verdict.correct,
                "judge_parsed": verdict.parsed,
            }
        rows.append(row)

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    answered = [r for r in answerable if not r["refused"]]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    latencies = [r["latency_ms"] for r in rows]
    input_tokens = sum(r["input_tokens"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in rows)
    cost = (
        input_tokens * settings.price_input_per_mtok
        + output_tokens * settings.price_output_per_mtok
    ) / 1_000_000

    return {
        "mode": mode.value,
        "metrics": {
            "answered_rate": rate([not r["refused"] for r in answerable]),
            # Correct answers among ALL answerable questions (a refusal counts as wrong).
            "correctness": rate([bool(r.get("correct")) for r in answerable]),
            "faithfulness": rate([r["faithful"] for r in answered]),
            "citation_accuracy": rate([r["citation_hit"] for r in answered]),
            "refusal_accuracy": rate([r["refused"] for r in unanswerable]),
            "judge_parse_failures": sum(1 for r in answered if not r["judge_parsed"]),
            "latency_p50_ms": round(percentile(latencies, 50), 1),
            "latency_p95_ms": round(percentile(latencies, 95), 1),
            "avg_input_tokens": round(input_tokens / len(rows), 1),
            "avg_output_tokens": round(output_tokens / len(rows), 1),
            "cost_per_question_usd": round(cost / len(rows), 6),
        },
        "per_question": rows,
    }


def save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_retrieval(report: dict) -> None:
    columns = ("Recall@1", 10), ("Recall@5", 10), ("MRR@5", 8), ("Rejected", 10), ("p50 ms", 9)
    print("\n" + "mode".ljust(15) + "".join(name.rjust(width) for name, width in columns))
    for mode, m in report["modes"].items():
        print(
            f"{mode:<15}{m['recall_at_1']:>10.1%}{m['recall_at_5']:>10.1%}"
            f"{m['mrr_at_5']:>8.3f}{m['unanswerable_rejected']:>10.1%}{m['latency_p50_ms']:>9.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run")
    parser.add_argument("target", choices=["retrieval", "answers"])
    parser.add_argument("--mode", choices=[m.value for m in SearchMode], default=None)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="quality gate: exit 1 if Recall@5 of the default search mode is lower",
    )
    args = parser.parse_args()

    items = load_questions()
    # ChromaDB keeps its files open until the process exits, which Windows won't delete.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        rag, settings = build_eval_service(Path(tmp))
        header = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": config_summary(settings),
        }

        if args.target == "retrieval":
            report = header | evaluate_retrieval(rag, items)
            path = save("retrieval", report)
            print_retrieval(report)
            print(f"\nsaved {path}")
            default_mode = SearchMode(settings.search_mode).value
            recall = report["modes"][default_mode]["recall_at_5"]
            if args.min_recall is not None and recall < args.min_recall:
                print(f"FAIL: {default_mode} Recall@5 {recall:.1%} is below {args.min_recall:.1%}")
                return 1
            return 0

        mode = SearchMode(args.mode or settings.search_mode)
        print(f"Answering {len(items)} questions with {settings.llm_model} ({mode})...")
        try:
            report = header | evaluate_answers(rag, settings, items, mode)
        except LLMError as exc:
            print(exc, file=sys.stderr)
            return 1
        path = save("answers", report)
        for key, value in report["metrics"].items():
            print(f"  {key:<24}{value}")
        print(f"\nsaved {path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
