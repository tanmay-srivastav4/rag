from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from dotenv import load_dotenv
import os
import uuid
import logging
import shutil

load_dotenv()
logging.basicConfig(filename='app.log', level=logging.INFO)

app = FastAPI()

BLOCKED_KEYWORDS = [
    "aws key", "aws keys", "access key", "secret key", "api key", "api keys",
    "password", "connection string", "postgresql",
    "private key", "ssh key", "aadhaar", "aadhar", "pan card", "pan number",
    "bank account", "ifsc", "salary", "ctc", "compensation",
    "employee details", "give me details", "all employees", "employee data",
    "show me employees", "list employees", "financial data", "cap table",
    "acquisition", "ipo", "legal case", "whistleblower",
]

BLOCKED_RESPONSE = "I am not authorised to share that information directly."

def is_simple_blocked_query(query: str) -> bool:
    query_lower = query.lower().strip()
    if len(query_lower.split()) <= 12:
        for keyword in BLOCKED_KEYWORDS:
            if keyword in query_lower:
                return True
    return False


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    if is_simple_blocked_query(query_input.question):
        logging.info(f"Session ID: {session_id}, Query blocked by keyword filter.")
        insert_application_logs(session_id, query_input.question, BLOCKED_RESPONSE, query_input.model.value)
        return QueryResponse(answer=BLOCKED_RESPONSE, session_id=session_id, model=query_input.model)

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {"message": f"File {file.filename} uploaded and indexed successfully.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id}."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete file_id {request.file_id} from database."}
    else:
        return {"error": f"Failed to delete file_id {request.file_id} from Chroma."}