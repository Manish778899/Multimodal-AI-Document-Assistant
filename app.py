import io
import os
import tempfile
from pathlib import Path

import chromadb
import fitz
import numpy as np
import requests
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

try:
    import pytesseract
except ImportError:
    pytesseract = None


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multimodal AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


# ---------------------------------------------------------
# Cached AI components
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """Load the embedding model once and reuse it across reruns."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_client():
    """Use an in-memory vector database for the current demo session."""
    return chromadb.Client()


embedding_model = load_embedding_model()
chroma_client = get_chroma_client()


# ---------------------------------------------------------
# Document extraction
# ---------------------------------------------------------

def ocr_image(image: Image.Image) -> str:
    """Extract visible text from an image when Tesseract is available."""
    if pytesseract is None:
        return ""

    try:
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""


def extract_pdf(uploaded_file) -> list[dict]:
    """
    Extract text page-by-page from a PDF.

    If a page contains almost no selectable text, render it as an image
    and try OCR. This helps with scanned PDFs.
    """
    pdf_bytes = uploaded_file.getvalue()
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            if len(text) < 30:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                scanned_text = ocr_image(image)

                if scanned_text:
                    text = scanned_text

            pages.append(
                {
                    "page": page_index + 1,
                    "text": text,
                }
            )

    return pages


def extract_image(uploaded_file) -> list[dict]:
    """Run OCR on an uploaded image."""
    image = Image.open(uploaded_file).convert("RGB")
    text = ocr_image(image)

    return [
        {
            "page": 1,
            "text": text,
        }
    ]


def extract_document(uploaded_file) -> list[dict]:
    """Select the correct parser based on the uploaded extension."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(uploaded_file)

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        return extract_image(uploaded_file)

    raise ValueError("Unsupported file type.")


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split long page text into overlapping chunks.

    Overlap helps preserve context when an important sentence falls
    near the boundary between two chunks.
    """
    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())

        if not text:
            continue

        start = 0
        chunk_number = 0

        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "page": page["page"],
                        "chunk_number": chunk_number,
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            start = max(0, end - CHUNK_OVERLAP)
            chunk_number += 1

    return chunks


# ---------------------------------------------------------
# Vector storage + retrieval
# ---------------------------------------------------------

def create_collection():
    """
    Recreate one clean Chroma collection for the current uploaded document.
    Keeping one document at a time makes the interview demo easy to explain.
    """
    collection_name = "document_demo"

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_document(filename: str, chunks: list[dict]):
    """Convert text chunks into embeddings and save them in ChromaDB."""
    if not chunks:
        return None

    collection = create_collection()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(f"chunk-{index}")
        metadatas.append(
            {
                "filename": filename,
                "page": int(chunk["page"]),
                "chunk_number": int(chunk["chunk_number"]),
            }
        )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


def retrieve_chunks(collection, question: str, top_k: int = TOP_K) -> list[dict]:
    """Find the chunks most semantically similar to the user's question."""
    if collection is None or collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        hits.append(
            {
                "text": document,
                "filename": metadata["filename"],
                "page": int(metadata["page"]),
                "score": round(1.0 - float(distance), 4),
            }
        )

    return hits


# ---------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------

def build_prompt(question: str, hits: list[dict]) -> str:
    """Build a grounded prompt using only retrieved document context."""
    context_blocks = []

    for index, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"[Source {index}: {hit['filename']}, page {hit['page']}]\n"
            f"{hit['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a document question-answering assistant.

Rules:
1. Answer only from the supplied document context.
2. Do not invent facts, dates, names, numbers, or clauses.
3. If the answer is not present, say:
   "The document does not contain enough information to answer this."
4. Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()


def ask_ollama(question: str, hits: list[dict]) -> tuple[str, bool]:
    """
    Ask a local Ollama model.

    If Ollama is not running, the application still demonstrates retrieval
    by showing the most relevant document passage.
    """
    if not hits:
        return "No relevant document content was found.", False

    prompt = build_prompt(question, hits)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()
        answer = (result.get("response") or "").strip()

        if answer:
            return answer, True

    except requests.RequestException:
        pass

    return (
        "Ollama is not running, so the LLM generation step is unavailable.\n\n"
        "Most relevant retrieved passage:\n\n"
        f"{hits[0]['text']}",
        False,
    )


# ---------------------------------------------------------
# Streamlit session state
# ---------------------------------------------------------

if "collection" not in st.session_state:
    st.session_state.collection = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("📄 Multimodal AI Document Assistant")
st.caption(
    "Upload a PDF or image, create embeddings, retrieve relevant content, "
    "and generate a grounded answer with page-level sources."
)

left, right = st.columns([0.42, 0.58], gap="large")


with left:
    st.subheader("1. Upload & Index")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        st.write(f"**Selected:** {uploaded_file.name}")

    if st.button("Index Document", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Choose a document first.")
        else:
            try:
                with st.spinner("Extracting text and creating embeddings..."):
                    pages = extract_document(uploaded_file)
                    chunks = chunk_pages(pages)

                    if not chunks:
                        st.error(
                            "No text could be extracted. "
                            "For scanned PDFs/images, install Tesseract OCR."
                        )
                    else:
                        collection = index_document(uploaded_file.name, chunks)

                        st.session_state.collection = collection
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.indexed_chunks = len(chunks)

                        st.success("Document indexed successfully.")

            except Exception as error:
                st.error(f"Could not process document: {error}")

    st.divider()

    st.subheader("Document Status")

    if st.session_state.document_name:
        st.metric("Indexed Chunks", st.session_state.indexed_chunks)
        st.write(f"**Document:** {st.session_state.document_name}")
    else:
        st.info("No document indexed yet.")

    st.divider()

    st.subheader("Pipeline")

    st.code(
        """PDF / Image
    ↓
Text Extraction / OCR
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Similarity Search
    ↓
LLM
    ↓
Answer + Sources""",
        language=None,
    )


with right:
    st.subheader("2. Ask the Document")

    question = st.text_area(
        "Question",
        placeholder="Example: What are the main findings in this document?",
        height=120,
    )

    if st.button("Ask Document", type="primary", use_container_width=True):
        if st.session_state.collection is None:
            st.warning("Index a document first.")

        elif not question.strip():
            st.warning("Enter a question first.")

        else:
            with st.spinner("Retrieving relevant context..."):
                hits = retrieve_chunks(
                    st.session_state.collection,
                    question.strip(),
                )

                answer, llm_used = ask_ollama(
                    question.strip(),
                    hits,
                )

            st.markdown("### Answer")
            st.write(answer)

            if llm_used:
                st.success(f"Generated with local LLM: {OLLAMA_MODEL}")
            else:
                st.info(
                    "Retrieval is working. Start Ollama to enable the generation step."
                )

            st.markdown("### Sources")

            if not hits:
                st.write("No matching source pages found.")
            else:
                seen = set()

                for hit in hits:
                    source_key = (hit["filename"], hit["page"])

                    if source_key in seen:
                        continue

                    seen.add(source_key)

                    with st.container(border=True):
                        st.write(
                            f"**{hit['filename']} — Page {hit['page']}**"
                        )
                        st.caption(f"Similarity score: {hit['score']}")
                        st.write(hit["text"][:500] + ("..." if len(hit["text"]) > 500 else ""))


st.divider()

st.subheader("How this project works")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.info("📄\n\n**Document**")
c2.info("🔎\n\n**Extract / OCR**")
c3.info("✂️\n\n**Chunking**")
c4.info("🧠\n\n**Embeddings**")
c5.info("🗃️\n\n**Vector Search**")
c6.info("💬\n\n**LLM Answer**")
