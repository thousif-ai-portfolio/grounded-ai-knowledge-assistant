# Search Methods for RAG

## Semantic search

Semantic search compares embedding vectors. Two texts with similar meaning have
vectors that point in a similar direction, which is measured with cosine
similarity. Semantic search finds paraphrases well, for example matching
"car" with "automobile", but it can miss exact terms such as error codes or
product names.

## Keyword search with BM25

BM25 is a ranking function that scores documents by how often the query words
appear in them, giving more weight to rare words and less weight to very long
documents. It is excellent at exact matches but does not understand synonyms.

## Hybrid search

Hybrid search runs semantic search and keyword search together and merges the
two result lists. A common merging method is Reciprocal Rank Fusion (RRF),
which adds 1 / (k + rank) for each list a passage appears in, usually with
k = 60. Hybrid search usually retrieves better passages than either method alone.

## Reranking

A reranker is a cross-encoder model that reads the question and a passage
together and outputs a relevance score. It is slower than vector search, so it
is only applied to the top 20 to 50 candidates, which it reorders so the most
relevant passages come first.
