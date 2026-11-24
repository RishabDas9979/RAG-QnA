"""Web UI: upload documents, index them, then chat with them or search by meaning.

Run:
    streamlit run app.py
"""
import html
import os
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
ALLOWED = {".txt", ".md", ".pdf"}
KEY_VAR = {"gemini": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
DOCS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="RAG From Scratch", page_icon="🔎", layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
# Colors use transparency so the page looks right in both light and dark themes.
st.markdown(
    """
<style>
:root { --accent: #f2611d; --accent-soft: rgba(242, 97, 29, .12); --line: rgba(128,128,128,.25); }
.block-container { padding-top: 2rem; max-width: 1100px; }
#MainMenu, footer { visibility: hidden; }

.hero h1 { font-size: 2.6rem; font-weight: 800; line-height: 1.05; margin: 0; letter-spacing: -.02em; }
.hero h1 span { color: var(--accent); }
.hero p { opacity: .75; font-size: 1.05rem; margin: .5rem 0 1.4rem; max-width: 46rem; }

.stat { border: 1px solid var(--line); border-radius: 14px; padding: .9rem 1.1rem; background: var(--accent-soft); }
.stat .n { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.stat .l { font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em; }

.pipe { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: 1.2rem 0; }
.pipe .step { border: 1px solid var(--line); border-radius: 999px; padding: .35rem .9rem; font-size: .9rem; }
.pipe .step.on { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.pipe .arrow { opacity: .45; }

.card { border: 1px solid var(--line); border-radius: 14px; padding: .9rem 1.1rem; margin-bottom: .8rem; }
.card .src { font-weight: 600; font-size: .95rem; }
.card .txt { opacity: .9; margin-top: .45rem; font-size: .95rem; white-space: pre-wrap; }
.bar { height: 6px; border-radius: 6px; background: var(--line); margin-top: .4rem; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); border-radius: 6px; }
.match { font-size: .8rem; opacity: .75; }

.chip { display: inline-block; border: 1px solid var(--line); background: var(--accent-soft);
        border-radius: 999px; padding: .1rem .7rem; font-size: .8rem; margin: .1rem .3rem .1rem 0; }
.empty { border: 1.5px dashed var(--line); border-radius: 18px; padding: 2rem; text-align: center; }
.empty h3 { margin: 0 0 .4rem; }
.empty ol { text-align: left; display: inline-block; margin: .6rem 0 0; opacity: .85; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ helpers
def esc(text: str) -> str:
    return html.escape(text or "")


def load_file(path: Path):
    """Load one file into LangChain documents, tagged with its filename."""
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def add_files(uploaded):
    """Save uploaded files to docs/ and add them to the index.

    Returns (added, skipped, failed) lists of file names / messages.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    added, skipped, failed, chunks = [], [], [], []

    for f in uploaded:
        name = Path(f.name).name
        path = DOCS_DIR / name
        if path.exists():
            skipped.append(name)
            continue
        path.write_bytes(f.getbuffer())
        try:
            file_chunks = splitter.split_documents(load_file(path))
            if not file_chunks:
                raise ValueError("no text found")
        except Exception as e:  # unreadable file, scanned PDF, bad encoding...
            path.unlink(missing_ok=True)
            failed.append(f"{name} ({e})")
            continue
        chunks.extend(file_chunks)
        added.append(name)

    if chunks:
        rag.add_chunks(chunks)
        st.cache_resource.clear()  # reload the index so new docs are searchable
    return added, skipped, failed


def remove_file(name: str):
    """Delete a document from disk and from the index."""
    try:
        rag.delete_source(get_store(), name)
    except Exception as e:
        st.session_state["notice"] = ([], [], [f"{name} ({e})"])
        return
    (DOCS_DIR / name).unlink(missing_ok=True)
    st.cache_resource.clear()


@st.cache_resource(show_spinner="Loading index...")
def get_store():
    return rag.load_index()


@st.cache_resource(show_spinner="Loading model...")
def get_chain():
    return rag.build_chain(get_store())


def indexed_files():
    return sorted(p.name for p in DOCS_DIR.iterdir() if p.suffix.lower() in ALLOWED)


def relevance(distance: float) -> float:
    """Turn a squared-L2 distance between unit vectors into a 0..1 match score."""
    return max(0.0, min(1.0, 1 - distance / 2))


def stat(col, number, label):
    col.markdown(
        f'<div class="stat"><div class="n">{esc(str(number))}</div>'
        f'<div class="l">{esc(label)}</div></div>',
        unsafe_allow_html=True,
    )


def pipeline(active=None):
    steps = ["Your documents", "Semantic search", "Relevant context", "LLM", "Answer"]
    parts = []
    for i, s in enumerate(steps):
        cls = "step on" if s == active else "step"
        parts.append(f'<span class="{cls}">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="arrow">→</span>')
    st.markdown(f'<div class="pipe">{"".join(parts)}</div>', unsafe_allow_html=True)


def show_passages(passages):
    for p in passages:
        st.markdown(
            f'<div class="card"><div class="src">📄 {esc(p["source"])}</div>'
            f'<div class="txt">{esc(p["text"])}</div></div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------ state
st.session_state.setdefault("chat", [])
st.session_state.setdefault("upload_round", 0)
files = indexed_files()
index_ready = bool(files) and os.path.exists(rag.INDEX_DIR)

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 📚 Documents")

    uploaded = st.file_uploader(
        "Add .txt, .md or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['upload_round']}",
    )
    if st.button(
        "Add to index", type="primary", disabled=not uploaded, use_container_width=True
    ):
        with st.spinner("Reading, chunking and embedding..."):
            st.session_state["notice"] = add_files(uploaded)
        st.session_state["upload_round"] += 1
        st.rerun()

    notice = st.session_state.pop("notice", None)
    if notice:
        added, skipped, failed = notice
        if added:
            st.success("Added: " + ", ".join(added))
        if skipped:
            st.warning("Already indexed, skipped: " + ", ".join(skipped))
        if failed:
            st.error("Could not read: " + "; ".join(failed))

    st.divider()
    st.markdown(f"**Indexed ({len(files)})**")
    if not files:
        st.caption("Nothing yet. Upload a file above.")
    for name in files:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f"📄 {name}")
        if rag.can_delete() and c2.button("", icon=":material/delete:", key=f"del_{name}", help=f"Remove {name}"):
            remove_file(name)
            st.rerun()

    st.divider()
    st.markdown("**Model**")
    st.markdown(
        f'<span class="chip">{esc(rag.LLM_PROVIDER)}</span>'
        f'<span class="chip">{esc(rag.LLM_MODEL)}</span>',
        unsafe_allow_html=True,
    )
    key_var = KEY_VAR.get(rag.LLM_PROVIDER)
    if key_var and not os.getenv(key_var):
        st.warning(f"`{key_var}` is not set in this terminal, so Ask will fail.")
    if st.session_state["chat"] and st.button("Clear chat", use_container_width=True):
        st.session_state["chat"] = []
        st.rerun()

# --------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><h1>RAG <span>From Scratch</span></h1>'
    "<p>Searches your documents by meaning, pulls the right passages, and hands "
    "them to the LLM before it answers. Fewer hallucinations, with sources.</p></div>",
    unsafe_allow_html=True,
)

if not index_ready:
    pipeline()
    st.markdown(
        '<div class="empty"><h3>No documents yet</h3>'
        "<div>Add a file to start asking questions about it.</div>"
        "<ol><li>Open the sidebar on the left</li>"
        "<li>Choose a .txt, .md or .pdf file</li>"
        "<li>Click <b>Add to index</b></li></ol></div>",
        unsafe_allow_html=True,
    )
    st.stop()

store = get_store()
c1, c2, c3 = st.columns(3)
stat(c1, len(files), "Documents")
stat(c2, rag.count_chunks(store), "Searchable chunks")
stat(c3, rag.TOP_K, "Passages per answer")
st.write("")

ask_tab, search_tab = st.tabs(["💬 Ask", "🔍 Semantic search"])

# ----------------------------------------------------------------- Ask
with ask_tab:
    if not st.session_state["chat"]:
        pipeline("Answer")
        st.caption("Ask a question about your documents below.")

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                st.markdown(
                    "".join(f'<span class="chip">📄 {esc(s)}</span>' for s in msg["sources"]),
                    unsafe_allow_html=True,
                )
                with st.expander("Passages used"):
                    show_passages(msg["passages"])

    question = st.chat_input("Ask something about your documents...")
    if question and question.strip():
        question = question.strip()
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and thinking..."):
                try:
                    result = get_chain().invoke(question)
                except Exception as e:
                    st.error(f"Could not get an answer: {e}")
                else:
                    st.session_state["chat"].append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "passages": result["passages"],
                        }
                    )
                    st.rerun()

# -------------------------------------------------------------- Search
with search_tab:
    st.caption(
        "Finds passages by meaning, not exact words. No LLM is involved, "
        "so this shows what the retrieval step hands to the model."
    )
    with st.form("search_form", border=False):
        sc1, sc2 = st.columns([4, 1], vertical_alignment="bottom")
        query = sc1.text_input("Search by meaning", placeholder="e.g. why do models make things up?")
        k = sc2.number_input("Results", 1, 10, 4)
        searched = st.form_submit_button("Search", type="primary")

    if searched and query.strip():
        with st.spinner("Searching..."):
            hits = store.similarity_search_with_score(query.strip(), k=int(k))
        if not hits:
            st.info("No matches found.")
        for doc, dist in hits:
            score = relevance(dist)
            st.markdown(
                f'<div class="card"><div class="src">📄 {esc(doc.metadata.get("source", "unknown"))}</div>'
                f'<div class="bar"><div style="width:{score * 100:.0f}%"></div></div>'
                f'<div class="match">{score * 100:.0f}% match</div>'
                f'<div class="txt">{esc(doc.page_content)}</div></div>',
                unsafe_allow_html=True,
            )
