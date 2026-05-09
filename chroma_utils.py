"""
chroma_utils.py
ChromaDB vector-store helpers.

Single source of truth for the text splitter, embedding model, and
vector-store configuration — ingest.py imports from here instead of
duplicating settings.
"""

import os
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Shared configuration — imported by ingest.py as well
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "arjuna_internal_docs"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", " "],
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_function,
    collection_name=COLLECTION_NAME,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vectorstore() -> Chroma:
    return vector_store


def load_and_split_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
    return text_splitter.split_documents(loader.load())


def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata["file_id"] = file_id
        vector_store.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def delete_doc_from_chroma(file_id: int) -> bool:
    try:
        docs = vector_store.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} chunk(s) for file_id {file_id}")
        vector_store._collection.delete(where={"file_id": file_id})
        print(f"Deleted all chunks with file_id {file_id}")
        return True
    except Exception as e:
        print(f"Error deleting document from Chroma: {e}")
        return False