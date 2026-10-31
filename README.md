# Grounded AI Knowledge Assistant

Ask questions about your PDFs and documents and get answers **with citations**
(file and page). When the documents don't contain the answer, it says
**"I don't know"** instead of guessing.

It runs entirely on your own machine for free: local models, no API keys.

![CI](https://github.com/thousif-ai-portfolio/grounded-ai-knowledge-assistant/actions/workflows/ci.yml/badge.svg)

**Video walkthrough (2.5 min):** [docs/demo.mp4](docs/demo.mp4) · narration in
[docs/video-script.md](docs/video-script.md)

## Features

- **Cited answers, streamed.** The answer appears as the model writes it. Every claim
  points to a source, and you can expand the citation to read the exact passage and page.
- **Hybrid search.** Vector search (meaning) and BM25 (exact words) are merged with
  Reciprocal Rank Fusion, then a cross-encoder reranks the results.
- **Refuses instead of guessing.** Off-topic questions are rejected before the model
  is called, and the model is told to answer "I don't know" when the sources are silent.
- **Measured, not claimed.** An 80-question benchmark (including misspelled questions)
  compares search strategies, and CI fails if search quality drops.
- **Full-stack app.** FastAPI backend, React frontend, user accounts with JWT, private
  chat history, and an evaluation dashboard.

## Results

Measured with `python -m evaluation.run retrieval` on 80 questions over 7 documents:
58 natural and keyword questions, 10 misspelled ones like *"wat does E1042 meen"*, and
12 whose answer is not in the documents. Each question is labelled with an evidence
phrase that must appear in the retrieved chunk.

| Search mode | Recall@1 | Recall@5 | MRR@5 | Recall@5, misspelled | Latency p50 |
|---|---:|---:|---:|---:|---:|
| Semantic (vectors only) | 77.9% | 95.6% | 0.855 | 70% | 30 ms |
| Hybrid (vectors + BM25, RRF) | 80.9% | 98.5% | 0.891 | 90% | 30 ms |
| **Hybrid + rerank** (default) | **95.6%** | 98.5% | **0.971** | 90% | 1,706 ms |

**What this shows**

- Every mode finds the right passage for all 58 correctly spelled questions. The
  difference is **ranking**: reranking puts the right passage *first* 95.6% of the time
  instead of 78%. That matters because a small model mostly answers from the first sources.
- **Misspellings are where vectors alone fall short** (70%): BM25 still matches the
  words that are spelled right. The one miss in every mode, *"wat is a contxt windo"*,
  has no correctly spelled key word left to match.
- Reranking costs about **2 seconds per question on a laptop CPU**. That's worth it
  here, because generating the answer takes longer anyway; for a latency-critical
  service, hybrid without reranking is the cheaper option.
- 5 of the 12 unanswerable questions (the clearly off-topic ones, like "What is the
  capital of France?") are rejected before the model is called. The other 7 are
  near-misses, like *"What does Atlas error E4040 mean?"* when only other codes exist.
  They look relevant to search, so refusing them is the model's job.

**Answer quality** is measured with `python -m evaluation.run answers` on the same
80 questions, using hybrid + rerank and `llama3.2:3b` both to answer and to judge
(laptop CPU, no GPU).

| Metric | Result |
|---|---:|
| Correctness (answerable questions) | 94.1% (64 / 68) |
| Faithfulness to the sources | 91.0% (61 / 67) |
| Citation accuracy | 100% (67 / 67) |
| Refusal accuracy (unanswerable questions) | 100% (12 / 12) |
| Latency p50 / p95 | 33.1 s / 52.3 s |
| Tokens per question (in / out) | 867 / 23 |
| Cost per question | $0 (runs locally) |

- **Every unanswerable question got "I don't know"**, including the 7 near-misses
  that got past the search cutoff, so the prompt handles what the cutoff can't.
- **9 of the 10 misspelled questions were answered correctly.**
- **Faithfulness is the weakest score, and partly noise.** The judge is the same small
  model, and grading the same saved answers twice gave 93.1% and 91.4%. Reading the
  flagged answers shows several are accurate paraphrases, so treat it as approximate.
- **Benchmarking found two bugs.** The model sometimes cites as `[1, 4]`, which the
  app didn't recognise, so those answers showed no sources. And when only an error
  code was asked, the model mixed in the fix for the code listed next to it; a prompt
  rule to answer only about the code asked fixed that, with correctness, citations
  and refusals unchanged on the original 70 questions.
- **Fixing the grader raised faithfulness from 89.7% to 94.8%** on the earlier
  70-question version of the benchmark. Reading the failures showed the judge, not
  the answers, was often wrong: it marked paraphrases as incorrect and true extra
  details from the sources as made up. Clearer judge instructions with worked
  examples fixed that, and letting a question accept more than one evidence passage
  fixed a citation that pointed to a second valid page (98.3% → 100%). The answers
  themselves didn't change.
- **Latency is almost all generation** on a laptop CPU; search and reranking take
  about 2 seconds of it.

## Architecture

```mermaid
flowchart LR
    subgraph Indexing
        F[PDF / MD / TXT] --> P[PyMuPDF<br/>text + page numbers]
        P --> C[Chunks<br/>800 chars, 150 overlap]
        C --> E[Embeddings<br/>all-MiniLM-L6-v2]
        E --> V[(ChromaDB)]
    end
    subgraph Answering
        Q[Question] --> S1[Vector search]
        Q --> S2[BM25 search]
        S1 --> G{Closest chunk similar enough,<br/>or a rare keyword matches?}
        G -- no --> IDK[I don't know]
        G -- yes --> RRF[Reciprocal Rank Fusion]
        S2 --> RRF
        RRF --> R[Cross-encoder rerank]
        R --> L[Ollama llama3.2<br/>answers from numbered sources]
        L --> A[Answer + citations]
    end
    V -.-> S1
```

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Pydantic |
| Documents | PyMuPDF |
| Embeddings / reranking | Sentence Transformers (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) |
| Search | ChromaDB (vectors), rank-bm25 (keywords) |
| Answers | Ollama + `llama3.2:3b` (any Ollama model via one setting) |
| Accounts | SQLite + SQLAlchemy, Argon2 password hashing, JWT |
| Frontend | React, Vite, TypeScript, Tailwind CSS, Recharts |
| Quality | pytest, Vitest + Testing Library, Ruff, oxlint, GitHub Actions |
| Deployment | Docker, docker compose, Hugging Face Spaces |

## Design decisions and trade-offs

- **No LangChain or LlamaIndex.** Chunking, retrieval, fusion and prompting are each
  a few dozen lines of plain Python, so every step can be explained, tested and
  measured on its own.
- **Chunks never cross page boundaries**, so every citation points to a real page.
- **Evaluation labels are evidence phrases, not chunk IDs.** Changing the chunk size
  doesn't invalidate the dataset, and a test checks that every phrase really exists
  in its source document.
- **The "I don't know" cutoff (`min_score = 0.2`) was chosen from data.** Off-topic
  questions score 0.07–0.15 and the weakest answerable question scores 0.24, but one
  unanswerable near-miss scores 0.25. No single cutoff separates them, so the cutoff
  only catches the clearly off-topic questions and the model handles the rest.
- **Misspellings get a second chance.** *"chunck overlap"* scores 0.14, as low as
  off-topic questions. Between 0.1 and 0.2 a question still passes if it shares a rare
  word with the documents (here "overlap"); the off-topic questions in that range share
  none. A reranker score didn't work for this: it scores misspellings as low as
  off-topic questions.
- **Reciprocal Rank Fusion instead of mixing scores.** Cosine similarity and BM25
  scores are on different scales; RRF only uses ranks, so it needs no tuning.
- **Shared knowledge base, private history.** Any signed-in user can add documents;
  chat history is per user. Per-user document collections would be the next step for
  a multi-tenant product.
- **`temperature: 0`** makes answers repeatable, so benchmark runs are comparable.

## Run it locally

You need Python 3.11+, Node 22+ and [Ollama](https://ollama.com).

```bash
ollama pull llama3.2:3b
```

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m app.cli ingest data/sample_docs
```

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn app.main:app --reload
```

Open http://localhost:8000, create an account and ask a question. The API docs are at
http://localhost:8000/docs.

On a laptop with 8 GB of RAM, `llama3.2:1b` is noticeably faster. Set
`RAG_LLM_MODEL=llama3.2:1b` in a `.env` file (see `.env.example` for every setting).

**Frontend development** with hot reload: run `npm run dev` in `frontend/` while the
API runs on port 8000.

### With Docker

```bash
docker compose up --build
```

This starts the app and an Ollama container, and downloads the answer model on first
start. Set `RAG_JWT_SECRET` to a long random value before exposing it anywhere.

## Run the benchmark

```bash
python -m evaluation.run retrieval
python -m evaluation.run answers
python -m evaluation.run regrade
```

The first command needs no LLM. The second needs Ollama running and takes about an
hour on a laptop CPU. The third grades the saved answers again without regenerating them, which
is useful after changing the judge. Results are saved to `evaluation/results/` and shown on the dashboard.
Questions live in `evaluation/questions.jsonl`.

## Tests

```bash
pytest
ruff check .
```

```bash
cd frontend
npm test
npm run lint
```

There are 145 backend tests and 14 frontend tests. They use a fake embedder and
model, so they run offline in seconds. CI also runs the retrieval benchmark with the
real models and **fails the build if Recall@5 drops below 90%**.

## API

| Method | Path | Auth | What it does |
|---|---|---|---|
| `POST` | `/api/auth/register` | | Create an account |
| `POST` | `/api/auth/login` | | Get a JWT (form fields `username`, `password`) |
| `GET` | `/api/auth/me` | ✔ | Current user |
| `POST` | `/api/documents` | ✔ | Upload a PDF, Markdown or text file |
| `GET` | `/api/documents` | ✔ | List documents |
| `DELETE` | `/api/documents/{doc_id}` | ✔ | Remove a document |
| `POST` | `/api/ask` | ✔ | `{"question": "...", "mode": "hybrid_rerank"}` returns an answer with citations |
| `POST` | `/api/ask/stream` | ✔ | Same input; streams the answer as newline-delimited JSON (`delta` lines, then one `answer` line). The chat page uses this |
| `GET` / `DELETE` | `/api/history` | ✔ | Your chat history |
| `GET` | `/api/evaluation` | | Benchmark results for the dashboard |

### Sample output

Real responses from `POST /api/ask` on a laptop CPU (citation text shortened):

```json
{
  "answer": "Error E1042: the local cache is full. Run atlas prune or increase the space available to cache_dir. [1]",
  "citations": [
    {
      "number": 1,
      "filename": "atlas-sync-manual.pdf",
      "page": 4,
      "text": "Error Codes E1001: the configuration file atlas.toml could n...",
      "score": 4.52
    }
  ],
  "grounded": true,
  "mode": "hybrid_rerank",
  "retrieval_ms": 4403.3,
  "generation_ms": 47675.7,
  "input_tokens": 621,
  "output_tokens": 27
}
```

Asked about an error code that is not in the manual (E4040), it refuses:

```json
{
  "answer": "I don't know.",
  "citations": [],
  "grounded": false,
  "mode": "hybrid_rerank",
  "retrieval_ms": 1717.9,
  "generation_ms": 11295.5,
  "input_tokens": 588,
  "output_tokens": 6
}
```

## Deploy

See [deploy/huggingface/README.md](deploy/huggingface/README.md) to host it on a
Hugging Face Space. Docker Spaces need a Hugging Face PRO account, so there is no live
demo; the [video walkthrough](docs/demo.mp4) shows the app running.

## Project structure

```
app/            FastAPI app: ingestion, retrieval, RAG, auth, database
evaluation/     80-question benchmark, metrics, LLM judge, saved results
frontend/       React + Vite + TypeScript app
tests/          Backend tests
data/sample_docs/  Sample documents (the Atlas Sync manual is fictional)
scripts/        Generates the sample PDF
deploy/         Hugging Face Space setup
docs/           Video walkthrough and its narration
```

## Limitations

- **The benchmark corpus is small** (7 documents, about 20 chunks), so Recall@5
  saturates at 100%. A larger corpus, for example public research papers, would show
  bigger differences between modes.
- **The `min_score` cutoff was tuned on the same questions it's evaluated on**, so
  the rejection rate is optimistic.
- **The answer judge is the same small model** that writes the answers; a stronger
  judge would give more reliable faithfulness scores.
- **Scanned PDFs aren't supported** (they need OCR).
- **Prompt injection is only partly handled.** The separate test harness
  [llm-eval-quality-gate](https://github.com/thousif-ai-portfolio/llm-eval-quality-gate)
  found two cases it gets wrong: it obeys an instruction inside the question to append a
  code, and it repeats a poisoned document's request to email a password. Treating
  retrieved text strictly as data, or an output filter, would fix both.
- **Heavily misspelled questions can still get "I don't know"** when no key word is
  spelled right, like *"wat is a contxt windo"*.
- **Answers take about 30 seconds** because the model runs on a CPU; a GPU or a hosted
  model would bring that down to a few seconds.
- **No live demo.** Hugging Face Docker Spaces need a PRO account, so the Space setup
  in `deploy/huggingface/` has not been deployed. The Docker image is built by CI on
  every push and was run locally: it started in 20 seconds and answered with a correct
  citation, using Ollama on the host. The bundled Ollama service in `docker-compose.yml`
  has not been run on this laptop (8 GB of RAM).
