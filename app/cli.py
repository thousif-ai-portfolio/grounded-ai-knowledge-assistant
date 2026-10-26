"""Use the assistant from the terminal, without starting the API.

    python -m app.cli ingest data/sample_docs
    python -m app.cli ask "What is chunk overlap?"
"""

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.ingest import SUPPORTED_EXTENSIONS, UnsupportedFileError
from app.llm import LLMError
from app.main import build_service
from app.rag import EmptyDocumentError


def _files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            matches = (p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
            files.extend(sorted(matches))
        else:
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Index files or folders")
    ingest.add_argument("paths", nargs="+")
    ask = commands.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    args = parser.parse_args()

    rag = build_service(get_settings())

    if args.command == "ingest":
        for path in _files(args.paths):
            try:
                added = rag.add_document(path.name, path.read_bytes())
            except (UnsupportedFileError, EmptyDocumentError) as exc:
                print(f"skipped {path}: {exc}")
                continue
            state = "already indexed" if added.already_indexed else f"{added.chunks} chunks"
            print(f"{path}: {state}")
        return 0

    try:
        answer = rag.ask(args.question)
    except LLMError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(answer.answer)
    for c in answer.citations:
        print(f"  [{c.number}] {c.filename}, page {c.page} (score {c.score:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
