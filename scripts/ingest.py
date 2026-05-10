from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from backend.chroma_utils import CHROMA_DIR, text_splitter, vector_store

DATA_DIR = Path("data")


def ingest() -> None:
    documents = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    for file_path in pdf_files:
        print(f"Loading: {file_path.name}")
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
        documents.extend(docs)

    print(f"Total pages loaded: {len(documents)}")
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    print(f"Done. Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()