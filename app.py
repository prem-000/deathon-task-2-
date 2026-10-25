import os
import hashlib
from io import BytesIO
from pathlib import Path

import chromadb
import pymupdf
import pytesseract
import streamlit as st
from PIL import Image
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
APP_TITLE = "DocuMind | AI Document Assistant"
PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / ".chroma_db"
COLLECTION_NAME = "documind_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_chroma_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# -----------------------------
# PDF extraction with OCR fallback
# -----------------------------
def extract_pdf_pages(pdf_bytes: bytes, filename: str):
    """
    Extract selectable text from each PDF page. If a page has little or no
    text, render it to an image and use Tesseract OCR.
    Returns records with source filename, 1-based page number and text.
    """
    extracted_pages = []

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if len(pdf) == 0:
                raise ValueError("The PDF contains no pages.")

            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                # OCR scanned pages and pages where the text layer is effectively empty.
                if len(text) < 25:
                    try:
                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2),
                            alpha=False,
                        )
                        image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
                        ocr_text = pytesseract.image_to_string(image).strip()
                        if len(ocr_text) > len(text):
                            text = ocr_text
                    except pytesseract.TesseractNotFoundError as exc:
                        raise RuntimeError(
                            "Tesseract OCR is not installed in this runtime. "
                            "In Colab, run: !apt-get -qq install -y tesseract-ocr"
                        ) from exc
                    except Exception as exc:
                        # Keep normal text if OCR fails for one page.
                        if not text:
                            st.warning(
                                f"OCR could not process page {page_index + 1} "
                                f"of {filename}: {exc}"
                            )

                if text:
                    extracted_pages.append(
                        {
                            "source": filename,
                            "page": page_index + 1,
                            "text": text,
                        }
                    )
    except Exception:
        raise

    return extracted_pages


# -----------------------------
# Indexing
# -----------------------------
def build_index(uploaded_files):
    if not uploaded_files:
        raise ValueError("Upload at least one PDF first.")

    page_records = []
    failed_files = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        try:
            pages = extract_pdf_pages(pdf_bytes, uploaded_file.name)
            if not pages:
                failed_files.append(
                    f"{uploaded_file.name}: no text found, even after OCR."
                )
            else:
                page_records.extend(pages)
        except Exception as exc:
            failed_files.append(f"{uploaded_file.name}: {exc}")

    if not page_records:
        detail = "\n".join(failed_files) if failed_files else "No readable PDF text found."
        raise ValueError(
            "No extractable text was found in the uploaded PDFs, even after OCR.\n\n"
            + detail
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for record in page_records:
        page_chunks = splitter.split_text(record["text"])
        for chunk_number, chunk_text in enumerate(page_chunks):
            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "source": record["source"],
                        "page": record["page"],
                        "chunk": chunk_number,
                    }
                )

    if not chunks:
        raise ValueError("Text was extracted, but no usable chunks were created.")

    embedding_model = load_embedding_model()
    texts = [item["text"] for item in chunks]
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    client = load_chroma_client()

    # Replace the old index only after extraction and embedding have succeeded.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    metadatas = []
    for i, item in enumerate(chunks):
        stable = hashlib.sha1(
            f"{item['source']}|{item['page']}|{item['chunk']}|{item['text']}".encode("utf-8")
        ).hexdigest()[:20]
        ids.append(f"chunk-{i}-{stable}")
        metadatas.append(
            {
                "source": item["source"],
                "page": int(item["page"]),
                "chunk": int(item["chunk"]),
            }
        )

    # Batch writes to avoid oversized Chroma requests on larger PDFs.
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    return len(page_records), len(chunks), len(failed_files), failed_files


# -----------------------------
# Retrieval and answer generation
# -----------------------------
def retrieve_passages(question: str, n_results: int):
    client = load_chroma_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    embedding_model = load_embedding_model()
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        passages.append(
            {
                "text": text,
                "source": metadata.get("source", "Unknown PDF"),
                "page": metadata.get("page", "?"),
                "distance": distance,
            }
        )
    return passages


def answer_question(question: str, passages):
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to Colab Secrets and set "
            "os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY') "
            "before starting Streamlit."
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    context_blocks = []
    for index, passage in enumerate(passages, start=1):
        context_blocks.append(
            f"[Source {index}: {passage['source']}, page {passage['page']}]\n"
            f"{passage['text']}"
        )

    prompt = f"""You are DocuMind, a careful document question-answering assistant.
Answer the user's question using ONLY the context below.
If the context does not contain the answer, say that the uploaded documents do not provide enough information.
Do not invent facts. Cite supporting passages inline using [filename, p. X].
Keep the answer clear and useful.

CONTEXT:
{chr(10).join(context_blocks)}

USER QUESTION:
{question}
"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return (getattr(response, "text", None) or "").strip()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b0d12; color: #eef2ff; }
    [data-testid="stSidebar"] { background: #11141c; border-right: 1px solid #252b38; }
    .hero {
        background: linear-gradient(135deg, #171d2b, #10131b);
        border: 1px solid #293244;
        border-radius: 22px;
        padding: 32px 36px;
        margin: 18px 0 24px 0;
    }
    .eyebrow { color: #8ea8ff; font-weight: 800; letter-spacing: 2px; font-size: 12px; }
    .hero h1 { font-size: 38px; margin: 16px 0; color: #f5f7ff; }
    .hero p { color: #aab5ca; font-size: 16px; }
    .stButton > button[kind="primary"] {
        background: #ff4b55; border: 0; color: white; border-radius: 10px;
        min-height: 44px; font-weight: 700;
    }
    .passage {
        background: #151a25; border: 1px solid #2a3243; border-radius: 12px;
        padding: 14px; margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_index_summary" not in st.session_state:
    st.session_state.last_index_summary = None

with st.sidebar:
    st.markdown("## Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Selectable-text PDFs and scanned PDFs are supported. Scanned pages are processed with OCR.",
    )

    retrieved_count = st.slider("Retrieved passages", min_value=2, max_value=10, value=4)

    if st.button("＋ Index documents", type="primary", use_container_width=True):
        try:
            with st.status("Extracting text, running OCR if needed, and building the index...", expanded=True) as status:
                page_count, chunk_count, failed_count, failed_files = build_index(uploaded_files)
                status.update(label="Index built successfully", state="complete", expanded=False)
            st.session_state.last_index_summary = {
                "pages": page_count,
                "chunks": chunk_count,
            }
            st.session_state.chat_history = []
            st.success(
                f"Indexed {page_count} pages into {chunk_count} chunks."
            )
            for failure in failed_files:
                st.warning(failure)
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    if st.button("Rebuild index", use_container_width=True):
        try:
            with st.status("Rebuilding the index...", expanded=True) as status:
                page_count, chunk_count, failed_count, failed_files = build_index(uploaded_files)
                status.update(label="Index rebuilt successfully", state="complete", expanded=False)
            st.session_state.last_index_summary = {
                "pages": page_count,
                "chunks": chunk_count,
            }
            st.session_state.chat_history = []
            st.success(f"Rebuilt index: {page_count} pages, {chunk_count} chunks.")
            for failure in failed_files:
                st.warning(failure)
        except Exception as exc:
            st.error(f"Rebuild failed: {exc}")

    st.divider()
    st.markdown("### Index status")
    try:
        current_collection = load_chroma_client().get_collection(COLLECTION_NAME)
        st.metric("Stored chunks", current_collection.count())
    except Exception:
        st.metric("Stored chunks", 0)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">DOCUMENT INTELLIGENCE · RAG + OCR</div>
      <h1>Ask your documents.</h1>
      <p>Upload PDFs, retrieve relevant passages, and get AI answers with source and page references.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    collection = load_chroma_client().get_collection(COLLECTION_NAME)
    has_index = collection.count() > 0
except Exception:
    has_index = False

if not has_index:
    st.info("Upload PDFs in the sidebar and click **Index documents** to get started. Scanned PDFs will be processed with OCR.")
else:
    st.caption("Ask a question about your indexed documents. Answers are grounded in retrieved passages.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved source passages"):
                    for passage in message["sources"]:
                        st.markdown(
                            f"**{passage['source']} · Page {passage['page']}**"
                        )
                        st.write(passage["text"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating an answer..."):
                    passages = retrieve_passages(question, retrieved_count)
                    if not passages:
                        answer = "I couldn't retrieve any passages. Please rebuild the index and try again."
                    else:
                        answer = answer_question(question, passages)
                st.markdown(answer or "I couldn't generate an answer for this question.")
                with st.expander("Retrieved source passages"):
                    for passage in passages:
                        st.markdown(
                            f"**{passage['source']} · Page {passage['page']}**"
                        )
                        st.write(passage["text"])
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer or "I couldn't generate an answer for this question.",
                        "sources": passages,
                    }
                )
            except Exception as exc:
                error_message = f"Unable to answer: {exc}"
                st.error(error_message)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_message}
                )
