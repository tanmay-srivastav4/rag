# Arjuna Knowledge Assistant — RAG Chatbot

Internal RAG (Retrieval-Augmented Generation) chatbot for Arjuna Technologies.  
Employees can ask natural-language questions about company documents; the system retrieves relevant passages and answers via Gemini.

---

## Architecture

```
streamlit_app.py        ← UI entry point
  sidebar.py            ← Model selector + document management (upload / list / delete)
  chat_interface.py     ← Chat history + input box

api_utils.py            ← HTTP calls to the FastAPI backend

main.py                 ← FastAPI backend (chat, upload, list, delete endpoints)
  langchain_utils.py    ← RAG chain (history-aware retriever + QA chain)
  chroma_utils.py       ← ChromaDB helpers (single source of config truth)
  db_utils.py           ← SQLite helpers (chat history + document metadata)
  pydantic_models.py    ← Shared request/response schemas

ingest.py               ← One-off bulk ingest of ./data/*.pdf into ChromaDB
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Ingest documents (first time, or when docs change)

Place PDFs in `./data/` then run:

```bash
python ingest.py
```

### 4. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

### 5. Start the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Supported file types

Upload via the sidebar (PDF, DOCX, HTML) or bulk-ingest PDFs via `ingest.py`.

---

## Security

Queries containing keywords related to credentials, PII, salary, or financial data are blocked at two layers:
1. **Backend keyword filter** (`main.py`) — fast, no LLM call needed.
2. **System prompt policy** (`langchain_utils.py`) — the LLM is instructed never to reveal sensitive data.

---

## Notes

- The SQLite database (`rag_app.db`) stores chat history and document metadata.
- ChromaDB persists embeddings in `./chroma_db/`.
- Both directories are created automatically on first run.