# RAG From Scratch

Search your documents by meaning, pull the right info, and hand it to the LLM
before it answers. Fewer hallucinations.

```
Your documents -> Semantic search -> Relevant context -> LLM -> Answer
```

## What you're building

1. Your documents, indexed so they're searchable (`ingest.py` or the web app)
2. The right context pulled in before the LLM writes anything (retriever in `rag.py`)
3. More accurate, source-cited answers that say "I don't know" when unsure
4. A retrieval flow that's ready to grow into production

## Tech stack

LangChain, ChromaDB or FAISS, sentence-transformers embeddings (run locally),
a retrieval chain, Streamlit for the UI, and your choice of LLM:
Google Gemini (free tier), Ollama (free, local) or Claude (paid API key).

## Setup

```bash
pip install -r requirements.txt
```

Then pick an LLM and set its key (see "Choosing an LLM" below).

## Run: web app (upload, search, ask)

```bash
streamlit run app.py
```

- **Upload** `.txt`, `.md` or `.pdf` files in the sidebar and click **Add to index**. Remove a document with its trash icon (Chroma only).
- **Ask** in a chat view. Each answer shows its source files and the exact passages used.
- **Semantic search** shows the closest passages for a query with a match percentage, without the LLM.
- The sidebar shows which model is active and warns if its API key isn't set.

## Run: command line

```bash
# 1. Put files in ./docs (a sample is included), then index them
python ingest.py

# 2. Ask questions
python rag.py "Why does RAG reduce hallucinations?"
python rag.py            # interactive mode
```

Note: `ingest.py` rebuilds the whole index from `./docs`. Stop the web app
before running it. For adding files without rebuilding, use the web app.

## Choosing an LLM

Set `LLM_PROVIDER` (default `gemini`):

| Provider | Cost | Setup |
| --- | --- | --- |
| `gemini` | Free tier | Get a key at https://aistudio.google.com/apikey, then set `GOOGLE_API_KEY` |
| `ollama` | Free, runs locally | Install Ollama, run `ollama pull llama3.2`, set `LLM_PROVIDER=ollama` |
| `anthropic` | Paid API credits | Create a key in the Claude Console, set `ANTHROPIC_API_KEY`, set `LLM_PROVIDER=anthropic` |

Windows PowerShell example (current window only):

```powershell
$env:GOOGLE_API_KEY="your_key_here"
```

To save a key permanently, run `setx GOOGLE_API_KEY "your_key_here"` and
restart the terminal or VS Code. Never put keys in code files or share them.

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` | `gemini`, `ollama` or `anthropic` |
| `LLM_MODEL` | provider default | Override the model name |
| `VECTOR_STORE` | `chroma` | `chroma` or `faiss` (use the same value everywhere) |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `TOP_K` | `4` | Chunks retrieved per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking for ingestion |
| `DOCS_DIR` / `INDEX_DIR` | `docs` / `index` | Input and index folders |

## How it works

- **Ingest:** load files, split into overlapping chunks, embed each chunk, store vectors on disk.
- **Retrieve:** embed the question, fetch the `TOP_K` nearest chunks.
- **Generate:** the prompt gives the LLM only those chunks and tells it to answer from them, or admit it doesn't know.
- **Cite:** each chunk carries its filename, so answers list their sources.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit web app: upload, semantic search, ask |
| `rag.py` | Embeddings, vector store, LLM selection, retrieval chain, CLI |
| `ingest.py` | Command-line indexing of everything in `./docs` |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | UI theme color (keep this folder next to `app.py`) |
| `docs/` | Your documents (includes `sample.md`) |

## Next steps toward production

Add reranking, hybrid (keyword + vector) search, an evaluation set of questions,
and an API (FastAPI) in front of `build_chain()`.
