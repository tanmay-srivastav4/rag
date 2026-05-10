import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
HEADERS = {"accept": "application/json", "Content-Type": "application/json"}


def get_api_response(question: str, session_id: str | None, model: str) -> dict | None:
    payload = {"question": question, "model": model}
    if session_id:
        payload["session_id"] = session_id

    try:
        response = requests.post(f"{API_BASE}/chat", headers=HEADERS, json=payload, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return None

    if response.status_code == 200:
        return response.json()

    st.error(f"Chat request failed ({response.status_code}): {response.text}")
    return None


def upload_document(file) -> dict | None:
    try:
        response = requests.post(
            f"{API_BASE}/upload-doc",
            files={"file": (file.name, file, file.type)},
            timeout=120,
        )
    except requests.RequestException as exc:
        st.error(f"Upload error: {exc}")
        return None

    if response.status_code == 200:
        return response.json()

    st.error(f"Upload failed ({response.status_code}): {response.text}")
    return None


def list_documents() -> list:
    try:
        response = requests.get(f"{API_BASE}/list-docs", timeout=15)
    except requests.RequestException as exc:
        st.error(f"Document list error: {exc}")
        return []

    if response.status_code == 200:
        return response.json()

    st.error(f"Could not fetch document list ({response.status_code}): {response.text}")
    return []


def delete_document(file_id: int) -> dict | None:
    try:
        response = requests.post(
            f"{API_BASE}/delete-doc",
            headers=HEADERS,
            json={"file_id": file_id},
            timeout=30,
        )
    except requests.RequestException as exc:
        st.error(f"Delete error: {exc}")
        return None

    if response.status_code == 200:
        return response.json()

    st.error(f"Delete failed ({response.status_code}): {response.text}")
    return None