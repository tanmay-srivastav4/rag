# Arjuna Knowledge Assistant

Internal Retrieval-Augmented Generation (RAG) chatbot for Arjuna Technologies. Employees can ask natural-language questions about company documents, and the app retrieves relevant passages before answering with Gemini.

## Project Structure

```text
backend/
  main.py               FastAPI app and API routes
  langchain_utils.py    RAG chain and prompt setup
  chroma_utils.py       ChromaDB loading, indexing, and deletion helpers
  db_utils.py           SQLite chat history and document metadata helpers
  pydantic_models.py    Request and response models

frontend/
  streamlit_app.py      Streamlit UI entry point
  sidebar.py            Model selector and document management UI
  chat_interface.py     Chat history and input UI
  api_utils.py          HTTP client helpers for the backend

scripts/
  ingest.py             Bulk-ingests PDFs from data/ into ChromaDB

data/                   Source documents for local ingestion
main.py                 Compatibility entry point for uvicorn
streamlit_app.py        Compatibility entry point for Streamlit
ingest.py               Compatibility entry point for ingestion
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

### 3. Ingest documents

Place PDFs in `data/`, then run:

```bash
python ingest.py
```

### 4. Start the backend

```bash
uvicorn main:app --reload
```

### 5. Start the frontend

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

## Supported Files

The sidebar upload supports PDF, DOCX, and HTML files. Bulk ingestion supports PDFs from `data/`.

## Runtime Files

The app creates local runtime files on first use:

- `rag_app.db` for chat history and document metadata
- `chroma_db/` for persisted vector embeddings
- `app.log` for backend logs