from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function,
    collection_name="arjuna_internal_docs"
)

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
    documents = loader.load()
    return text_splitter.split_documents(documents)

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
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        vector_store._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        return True
    except Exception as e:
        print(f"Error deleting document from Chroma: {e}")
        return False