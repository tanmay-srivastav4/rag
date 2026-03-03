import sqlite3
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage


DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        model TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                 )
                 ''')
    conn.commit()
    conn.close()

def create_document_store():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

#chat history functions
def insert_application_logs(session_id, user_query, response, model):
    conn = get_db_connection()
    conn.execute('''INSERT INTO application_logs (session_id, user_query, response, model)
                 VALUES (?, ?, ?, ?)''', (session_id, user_query, response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, response FROM application_logs WHERE session_id = ? ORDER BY timestamp ASC', (session_id,))
    chat_history = []
    for row in cursor.fetchall():
        chat_history.append([
            {"role": "human", "content":row["user_query"]},
            {"role": "assistant", "content":row["response"]}
        ])
    conn.close()
    return chat_history

#document store functions
def insert_document_record(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_query, response FROM application_logs WHERE session_id = ? ORDER BY timestamp ASC',
        (session_id,)
    )

    chat_history = []

    for row in cursor.fetchall():
        chat_history.append(HumanMessage(content=row["user_query"]))
        chat_history.append(AIMessage(content=row["response"]))

    conn.close()
    return chat_history

create_application_logs()
create_document_store()