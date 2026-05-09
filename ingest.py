"""
ingest.py
Bulk-ingest all PDFs from ./data into ChromaDB.

Run once (or whenever documents change):
    python ingest.py

Fixes vs original:
- No longer duplicates text_splitter / embedding_function / CHROMA_DIR /
  COLLECTION_NAME — all imported from chroma_utils.py (single source of truth).
"""

import os

from langchain_community.document_loaders import PyPDFLoader

from chroma_utils import (
    CHROMA_DIR,
    COLLECTION_NAME,
    embedding_function,
    text_splitter,
    vector_store,
)

DATA_DIR = "./data"


def ingest() -> None:
    documents = []
    pdf_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    for filename in pdf_files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Loading: {filename}")
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        documents.extend(docs)

    print(f"\nTotal pages loaded : {len(documents)}")
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    print(f"\n✅ Done — {len(chunks)} chunks stored in ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()