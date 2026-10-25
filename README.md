# DocuMind - AI Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) app for asking questions about PDF documents.

## Features
- PDF text extraction with OCR fallback for scanned PDFs
- Text chunking and semantic embeddings
- ChromaDB vector search
- Gemini-powered question answering
- Source and page citations

## Tech Stack
- Python, Streamlit
- PyMuPDF, Tesseract OCR
- Sentence Transformers
- ChromaDB
- Google Gemini API

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Install Tesseract OCR if scanned PDF support is needed.
3. Copy `.env.example` to `.env`.
4. Add your Gemini API key to `.env`.
5. Run: `streamlit run app.py`

## Environment variables
Set `GEMINI_API_KEY` to your own API key.
Set `GEMINI_MODEL` to a model available to your Gemini API account.

**Security:** Never commit `.env`, API keys, uploaded PDFs, or the local vector database.
