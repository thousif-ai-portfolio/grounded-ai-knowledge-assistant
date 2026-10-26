# Evaluating a RAG System

## Retrieval metrics

Recall@k is the fraction of questions for which at least one correct passage
appears in the top k retrieved results. Recall@5 is the most common version.
Mean Reciprocal Rank (MRR) averages 1 / rank of the first correct passage, so it
rewards systems that put the right passage near the top.

## Answer metrics

Faithfulness measures whether every claim in an answer is supported by the
retrieved passages. It is often scored by a second language model acting as a
judge. Answer correctness compares the answer with a reference answer written
by a person.

## Refusal cases

A good evaluation set includes questions whose answer is not in the documents.
For these, the correct response is "I don't know", and the system is scored on
how often it refuses instead of inventing an answer.

## Operational metrics

Latency is the time from receiving a question to returning an answer, usually
reported as the median (p50) and the 95th percentile (p95). Cost is the money
spent per question; with a local model it is zero, but the number of tokens
processed is still worth tracking.
