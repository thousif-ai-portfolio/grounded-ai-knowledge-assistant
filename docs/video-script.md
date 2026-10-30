# Video script: Grounded AI Knowledge Assistant

Narration for `docs/demo.mp4`, one section per scene, so it can be re-recorded.

## 1. Title

> Chatbots answer confidently even when they are wrong. I built an assistant that only answers from your own documents, cites the exact file and page, and says I don't know when the answer isn't there. It runs fully on my laptop: local models, no API keys, no cost.

## 2. Signed in: a private chat over the knowledge base

> This is the app. Every user has an account and a private chat history.

## 3. The knowledge base: PDF, Markdown and text files

> Here is the knowledge base: seven documents, including a five page PDF manual for a fictional product. You can upload PDFs, Markdown or text files. Each one is split into chunks that never cross a page, so every citation points to a real page.

## 4. Hybrid search, reranking, then a grounded answer

> A question is searched two ways at once: vector search for meaning, and BM25 for exact words like an error code. The two lists are fused, and a cross-encoder reranks the top results. The model then answers only from the numbered sources it was given.

## 5. Every sentence cited; the passage opens on click

> Here I asked what error E1042 means. The answer streams in as the model writes it, and every sentence is cited. Opening the citation shows the exact passage, on page four of the PDF.

## 6. Refusing instead of guessing

> Off-topic questions, like the capital of France, are rejected before the model is even called. The second one is harder: error E4040 sounds like it belongs in the manual, but it doesn't exist. The model is told to refuse when the sources are silent, and it does.

## 7. Misspelled on purpose: still the right answer

> Real users misspell things. Chunck overlap still gets the right answer, because the correctly spelled rare word, overlap, still matches.

## 8. The evaluation dashboard, from the saved benchmark

> I didn't want to just claim it works, so I built an eighty question benchmark, including trick questions and misspelled ones. The dashboard shows the saved results.

## 9. Measured on 80 questions

> Reranking puts the right passage first ninety five point six percent of the time, against seventy eight percent for vector search alone. Full answers are ninety four percent correct, every citation is right, and it refused all twelve trick questions. CI re-runs the search benchmark on every push and fails the build if quality drops.

## 10. Title

> It is written without LangChain, so every step is plain Python that I can explain and test. The code, the benchmark and every result are on GitHub. Thanks for watching.
