# Retrieval-Augmented Generation (RAG) Basics

Retrieval-augmented generation is a way to make a language model answer questions
about documents it was never trained on. Instead of relying on what the model
memorised, the system first retrieves relevant passages and then asks the model
to answer using only those passages.

## The pipeline

A RAG system has two phases. In the indexing phase, documents are split into
chunks, each chunk is converted into an embedding vector, and the vectors are
stored in a vector database. In the query phase, the user's question is embedded
with the same model, the closest chunks are retrieved, and the language model
writes an answer from them.

## Chunking

Chunk size controls how much text goes into each piece. Small chunks are precise
but can lose context; large chunks keep context but dilute relevance. Chunk
overlap repeats a few hundred characters between neighbouring chunks so that a
sentence cut at a boundary still appears complete in at least one chunk.

## Citations and refusals

A grounded answer cites the passages it used, so a reader can check every claim.
When no retrieved passage contains the answer, a well-behaved RAG system says
"I don't know" instead of guessing. Answers that state facts not found in the
sources are called hallucinations.
