# Grounded AI Knowledge Assistant

Ask questions about your PDFs and documents and get answers **with citations**
(file and page). If the documents don't contain the answer, it says
**"I don't know"** instead of guessing.

Everything runs locally and costs nothing: no API keys, no cloud.

> **Status: Week 1 of 5.** Core RAG pipeline + API + tests. Hybrid search,
> reranking, evaluation, UI and deployment are coming next (see [Roadmap](#roadmap)).

## How it works

```
Indexing:   PDF / .md / .txt ─► PyMuPDF (text + page numbers) ─► chunks (800 chars, 150 overlap)
                              ─► Sentence Transformers (all-MiniLM-L6-v2) ─► ChromaDB

Asking:     question ─► embed ─► ChromaDB top-5 ─► drop chunks below min_score
                     ─► Ollama (llama3.2:3b) answers from numbered sources only
                     ─► citations parsed from [1], [2] markers
```

Design decisions:

- **No LangChain / LlamaIndex.** Every step is plain Python, so each one can be
  explained, tested and swapped.
- **Chunks never cross pages**, so every citation points to a real page.
- **Two "I don't know" guards:** if no chunk passes `min_score`, the model is never
  called; if the model itself says it doesn't know, no citations are returned.
- **The model is one setting.** `RAG_LLM_PROVIDER` / `RAG_LLM_MODEL` switch it
  without touching the pipeline (see [app/llm.py](app/llm.py)).
- **Duplicate uploads are ignored.** A document's ID is a hash of its bytes.
- **`temperature: 0`**, so the same question gives the same answer, which keeps
  evaluation results repeatable.

## Tech stack

| Part | Tool |
|---|---|
| API | FastAPI |
| PDF text | PyMuPDF |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector database | ChromaDB (saved to `data/index/`) |
| Answer model | Ollama + `llama3.2:3b` |
| Tests / lint | pytest, Ruff |

## Setup (Windows, PowerShell)

1. Install **Python 3.11+** and **Ollama** (https://ollama.com).
2. Download the answer model (about 2 GB, one time):

   ```bash
   ollama pull llama3.2:3b
   ```

3. Create the virtual environment and install packages:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

The embedding model (about 90 MB) downloads automatically the first time you run the app.

## Run it

**From the terminal:**

```bash
python -m app.cli ingest data/sample_docs
python -m app.cli ask "What is chunk overlap used for?"
```

**As an API:**

```bash
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs to try every endpoint in the browser.

| Method | Path | What it does |
|---|---|---|
| `POST` | `/documents` | Upload a `.pdf`, `.md` or `.txt` file |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document |
| `POST` | `/ask` | `{"question": "...", "top_k": 5}` returns an answer with citations |
| `GET` | `/health` | Health check |

Example `/ask` response:

```json
{
  "answer": "Chunk overlap repeats text between neighbouring chunks so a sentence cut at a boundary still appears complete [1].",
  "citations": [
    {"number": 1, "filename": "rag-basics.md", "page": 1, "text": "...", "score": 0.71}
  ],
  "grounded": true
}
```

## Tests

```bash
pytest
ruff check .
```

Tests use a fake embedder and a fake model, so they run in seconds, offline,
without Ollama or any model downloads.

## Settings

Copy `.env.example` to `.env` and change any value, e.g. `RAG_LLM_MODEL=qwen2.5:3b`.

## Roadmap

| Week | Adds |
|---|---|
| 1 ✅ | PDF ingestion, semantic search, cited answers, "I don't know", API, tests |
| 2 | BM25 keyword search, hybrid search (Reciprocal Rank Fusion), cross-encoder reranking |
| 3 | 50–100 question evaluation set; Recall@5, MRR, faithfulness, latency; compare search modes |
| 4 | React + Vite + TypeScript chat UI, evaluation dashboard, SQLite + JWT login |
| 5 | Docker, GitHub Actions, deployment to Hugging Face Spaces, demo video |

## Limitations (current)

- Scanned PDFs (images only) aren't supported yet; they need OCR.
- A 3B local model is fast and free but less accurate than large hosted models.
- Semantic search only, so exact terms like error codes can be missed (fixed by BM25 in week 2).
