# Embeddings and Vector Search

## What an embedding is

An embedding model maps a piece of text to a fixed-length list of numbers called
a vector. The model all-MiniLM-L6-v2 produces vectors with 384 dimensions and was
trained on over one billion sentence pairs. Texts with similar meaning are mapped
to vectors that point in similar directions.

## Normalisation

Vectors are usually normalised to length one. For normalised vectors, cosine
similarity and dot product give the same ranking, and squared Euclidean distance
equals two minus two times the cosine similarity.

## Approximate nearest neighbour search

Comparing a question with every stored vector is called brute-force or exact
search, and it becomes slow for millions of vectors. Approximate nearest
neighbour (ANN) indexes trade a small amount of accuracy for large speed gains.
HNSW, short for Hierarchical Navigable Small World, builds a layered graph in
which each vector links to its close neighbours; a search starts at the top layer
and moves down. HNSW is the default index in ChromaDB.

## Vector databases

A vector database stores vectors together with metadata such as the file name
and page number, so results can be filtered and cited.

## Limitations

Embedding models have a maximum input length: all-MiniLM-L6-v2 truncates input
after 256 word pieces, which is one reason documents are split into chunks.
Embeddings also struggle with rare identifiers such as part numbers, which
keyword search handles better.
