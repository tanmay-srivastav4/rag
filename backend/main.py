import logging
import os
import shutil
import tempfile
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.chroma_utils import delete_doc_from_chroma, index_document_to_chroma
from backend.db_utils import (
    delete_document_record,
    get_all_documents,
    get_chat_history,
    insert_application_logs,
    insert_document_record,
)
from backend.langchain_utils import get_rag_chain
from backend.pydantic_models import DeleteFileRequest, DocumentInfo, QueryInput, QueryResponse

load_dotenv()
logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Arjuna Knowledge Assistant")

BLOCKED_KEYWORDS = [
    "aws key",
    "aws keys",
    "access key",
    "secret key",
    "api key",
    "api keys",
    "password",
    "connection string",
    "postgresql",
    "private key",
    "ssh key",
    "aadhaar",
    "aadhar",
    "pan card",
    "pan number",
    "bank account",
    "ifsc",
    "salary",
    "ctc",
    "compensation",
    "employee details",
    "give me details",
    "all employees",
    "employee data",
    "show me employees",
    "list employees",
    "financial data",
    "cap table",
    "acquisition",
    "ipo",
    "legal case",
    "whistleblower",
]

BLOCKED_RESPONSE = "I am not authorised to share that information directly."


def is_simple_blocked_query(query: str) -> bool:
    query_lower = query.lower().strip()
    return len(query_lower.split()) <= 12 and any(
        keyword in query_lower for keyword in BLOCKED_KEYWORDS
    )


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logger.info(
        "Session ID: %s, user query: %s, model: %s",
        session_id,
        query_input.question,
        query_input.model.value,
    )

    if is_simple_blocked_query(query_input.question):
        insert_application_logs(
            session_id,
            query_input.question,
            BLOCKED_RESPONSE,
            query_input.model.value,
        )
        return QueryResponse(
            answer=BLOCKED_RESPONSE,
            session_id=session_id,
            model=query_input.model,
        )

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke(
        {
            "input": query_input.question,
            "chat_history": chat_history,
        }
    )["answer"]

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logger.info("Session ID: %s, response generated", session_id)
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = [".pdf", ".docx", ".html"]
    filename = file.filename or "upload"
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        file_id = insert_document_record(filename)
        if index_document_to_chroma(temp_file_path, file_id):
            return {
                "message": f"File {filename} uploaded and indexed successfully.",
                "file_id": file_id,
            }

        delete_document_record(file_id)
        raise HTTPException(status_code=500, detail=f"Failed to index {filename}.")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    if not delete_doc_from_chroma(request.file_id):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file_id {request.file_id} from ChromaDB.",
        )
    if not delete_document_record(request.file_id):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Deleted from ChromaDB but failed to remove file_id "
                f"{request.file_id} from database."
            ),
        )

    logger.info("Deleted file_id=%s", request.file_id)
    return {"message": f"Document {request.file_id} deleted successfully."}