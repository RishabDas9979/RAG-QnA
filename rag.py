"""RAG from scratch: documents -> semantic search -> relevant context -> LLM -> answer.

Shared pieces (embeddings, vector store, retrieval chain) live here.
`ingest.py` builds the index; this file answers questions against it.

Usage:
    python rag.py "What does the document say about X?"
    python rag.py            # interactive mode
"""
import os
import sys

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------- config
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma").lower()  # "chroma" or "faiss"
INDEX_DIR = os.getenv("INDEX_DIR", "index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

# Which LLM answers the question: "gemini" (free tier), "ollama" (free, local),
# or "anthropic" (paid API key).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
    "anthropic": "claude-sonnet-5-5",
}
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(LLM_PROVIDER, ""))

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the context below. "
            "If the context does not contain the answer, say you don't know "
            "instead of guessing. Cite sources as [source].\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# ------------------------------------------------------- building blocks
def get_embeddings() -> HuggingFaceEmbeddings:
    """Embedding model: turns text into vectors so we can search by meaning."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm():
    """Create the chat model for the chosen provider."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Reads GOOGLE_API_KEY from the environment
        return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=LLM_MODEL, temperature=0)
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Reads ANTHROPIC_API_KEY from the environment
        return ChatAnthropic(model=LLM_MODEL)
    raise SystemExit(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use gemini, ollama or anthropic."
    )


def build_index(chunks):
    """Embed the chunks and persist them in the chosen vector store."""
    embeddings = get_embeddings()
    if VECTOR_STORE == "faiss":
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(INDEX_DIR)
    else:
        store = Chroma.from_documents(
            chunks, embeddings, persist_directory=INDEX_DIR
        )
    return store


def add_chunks(chunks):
    """Add new chunks to the existing index (creating it if needed)."""
    embeddings = get_embeddings()
    if VECTOR_STORE == "faiss":
        from langchain_community.vectorstores import FAISS

        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            store = FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
            store.add_documents(chunks)
        else:
            store = FAISS.from_documents(chunks, embeddings)
        store.save_local(INDEX_DIR)
    else:
        store = Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)
        store.add_documents(chunks)
    return store


def load_index():
    """Load a previously built index from disk."""
    if not os.path.exists(INDEX_DIR):
        raise SystemExit(
            f"No index found in '{INDEX_DIR}/'. Run `python ingest.py` first."
        )
    embeddings = get_embeddings()
    if VECTOR_STORE == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)


def count_chunks(store) -> int:
    """Number of chunks currently stored in the index."""
    if VECTOR_STORE == "faiss":
        return store.index.ntotal
    return store._collection.count()


def can_delete() -> bool:
    """Removing a single document is supported for Chroma only."""
    return VECTOR_STORE != "faiss"


def delete_source(store, source: str) -> None:
    """Remove every chunk that came from the given file name."""
    store._collection.delete(where={"source": source})


def format_docs(docs) -> str:
    """Join retrieved chunks into one context block, tagged with their source."""
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


def build_chain(store=None):
    """Retrieval chain: question -> retrieve -> prompt -> LLM -> answer.

    Pass an already-loaded vector store to reuse it; otherwise it is loaded
    from disk. Returns a runnable that outputs
    {"answer": str, "sources": [str, ...], "passages": [{"source", "text"}, ...]}.
    """
    if store is None:
        store = load_index()
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()

    answer_chain = PROMPT | llm | StrOutputParser()

    def run(question: str) -> dict:
        docs = retriever.invoke(question)
        answer = answer_chain.invoke(
            {"context": format_docs(docs), "question": question}
        )
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        passages = [
            {"source": d.metadata.get("source", "unknown"), "text": d.page_content}
            for d in docs
        ]
        return {"answer": answer, "sources": sources, "passages": passages}

    return RunnableLambda(run)


# ------------------------------------------------------------------ CLI
def main():
    chain = build_chain()

    def ask(q: str):
        result = chain.invoke(q)
        print(f"\n{result['answer']}\n")
        print("Sources:", ", ".join(result["sources"]), "\n")

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return

    print("RAG ready. Ask a question (empty line to quit).")
    while True:
        q = input("> ").strip()
        if not q:
            break
        ask(q)


if __name__ == "__main__":
    main()
