# Retrieval-Augmented Generation (RAG)

RAG combines a retriever with a language model. Instead of relying only on what
the model memorized during training, the system first searches a knowledge base
for passages relevant to the question, then gives those passages to the model as
context.

## Why it reduces hallucinations

Models tend to invent plausible-sounding answers when they lack facts. By
grounding the answer in retrieved text and instructing the model to say "I don't
know" when the context is silent, RAG keeps responses tied to real documents.

## Semantic search

Documents are split into chunks and converted into embedding vectors. A question
is embedded the same way, and the chunks whose vectors are closest to the
question vector are returned. This matches by meaning rather than exact keywords.

## Production tips

- Tune chunk size and overlap for your content.
- Store source metadata so answers can cite where they came from.
- Re-index whenever your documents change.
