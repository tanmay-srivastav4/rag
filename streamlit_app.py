"""
streamlit_app.py
Entry point for the Arjuna RAG Chatbot Streamlit application.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from chat_interface import display_chat_interface
from sidebar import display_sidebar

st.set_page_config(
    page_title="Arjuna Knowledge Assistant",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ Arjuna Knowledge Assistant")
st.caption("Ask questions about internal company documents. Sensitive data is protected.")

# Session state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "model" not in st.session_state:
    st.session_state.model = "gemini-2.5-flash"

display_sidebar()
display_chat_interface()