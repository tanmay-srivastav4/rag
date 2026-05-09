"""
api_utils.py
HTTP helpers that call the FastAPI backend from Streamlit.

Fixes vs original:
- Base URL is a single constant — change one line to switch environments.
- Errors surface via st.error (same as before) but the functions always return
  a consistent type (dict | None or list) so callers don't need to guard.
"""

import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"
_HEADERS = {"accept": "application/json", "Content-Type": "application/json"}


def get_api_response(question: str, session_id: str | None, model: str) -> dict | None:
    payload = {"question": question, "model": model}
    if session_id:
        payload["session_id"] = session_id
    try:
        r = requests.post(f"{API_BASE}/chat", headers=_HEADERS, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"Chat request failed ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Could not reach the API: {exc}")
    return None


def upload_document(file) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/upload-doc",
            files={"file": (file.name, file, file.type)},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"Upload failed ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Upload error: {exc}")
    return None


def list_documents() -> list:
    try:
        r = requests.get(f"{API_BASE}/list-docs", timeout=15)
        if r.status_code == 200:
            return r.json()
        st.error(f"Could not fetch document list ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Document list error: {exc}")
    return []


def delete_document(file_id: int) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/delete-doc",
            headers=_HEADERS,
            json={"file_id": file_id},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"Delete failed ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Delete error: {exc}")
    return None