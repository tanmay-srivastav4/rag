import sqlite3
from contextlib import contextmanager
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage

DB_NAME = "rag_app.db"


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_application_logs() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS application_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def create_document_store() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def insert_application_logs(
    session_id: str,
    user_query: str,
    response: str,
    model: str,
) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO application_logs (session_id, user_query, response, model) VALUES (?, ?, ?, ?)",
            (session_id, user_query, response, model),
        )


def get_chat_history(session_id: str) -> list[HumanMessage | AIMessage]:
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT user_query, response FROM application_logs "
            "WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )
        history: list[HumanMessage | AIMessage] = []
        for row in cursor.fetchall():
            history.append(HumanMessage(content=row["user_query"]))
            history.append(AIMessage(content=row["response"]))
        return history


def insert_document_record(filename: str) -> int:
    with get_db_connection() as conn:
        cursor = conn.execute("INSERT INTO document_store (filename) VALUES (?)", (filename,))
        return cursor.lastrowid


def delete_document_record(file_id: int) -> bool:
    with get_db_connection() as conn:
        conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
    return True


def get_all_documents() -> list[dict]:
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT id, filename, upload_timestamp FROM document_store "
            "ORDER BY upload_timestamp DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


create_application_logs()
create_document_store()