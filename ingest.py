import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "arjuna_internal_docs"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest():
    documents = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"Loading: {filename}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)

    print(f"\nTotal pages loaded: {len(documents)}")
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    print(f"\n✅ Done. {len(chunks)} chunks stored in ChromaDB at {CHROMA_DIR}")

if __name__ == "__main__":
    ingest()