"""Index your documents so they're searchable.

Put .txt, .md or .pdf files in ./docs, then run:
    python ingest.py
"""
import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            continue
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def main():
    if not DOCS_DIR.exists():
        raise SystemExit(f"Create a '{DOCS_DIR}/' folder and add some documents.")

    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .txt/.md/.pdf files found in '{DOCS_DIR}/'.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # Rebuild from scratch so the index always matches ./docs
    if os.path.exists(rag.INDEX_DIR):
        shutil.rmtree(rag.INDEX_DIR)

    rag.build_index(chunks)
    print(
        f"Indexed {len(docs)} document(s) as {len(chunks)} chunks "
        f"into '{rag.INDEX_DIR}/' ({rag.VECTOR_STORE})."
    )


if __name__ == "__main__":
    main()
