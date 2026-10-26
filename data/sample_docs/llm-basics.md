# Large Language Model Basics

## Tokens

Language models read and write text as tokens, which are pieces of words. In
English, one token is roughly four characters, or about three quarters of a word.

## Context window

The context window is the maximum number of tokens the model can read in one
request, including the prompt, the retrieved passages and the answer. Passages
that do not fit must be dropped, which is why retrieving the right passages
matters more than retrieving many of them.

## Temperature

Temperature controls randomness. A temperature of 0 makes the model pick the most
likely token each time, giving repeatable answers, which is useful for
evaluation. Higher temperatures produce more varied text.

## Hallucination

A hallucination is a fluent statement that is not supported by the sources or by
reality. Grounding answers in retrieved passages and requiring citations reduces
hallucinations but does not remove them.

## Prompt injection

Prompt injection happens when text inside a document or user message tries to
override the system's instructions, for example by saying "ignore previous
instructions". RAG systems are exposed because they paste document text into the
prompt. Mitigations include treating retrieved text as data, limiting which tools
the model can call, and testing with adversarial examples.

## Small local models

Models with about 3 billion parameters, such as Llama 3.2 3B, run on a laptop CPU
with around 4 GB of memory, but they follow complex instructions less reliably
than large hosted models.
