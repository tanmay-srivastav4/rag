"""
sidebar.py
Streamlit sidebar: model selection + full document management.

Improvements vs original:
- Original only had a model selector — document upload/list/delete were
  completely missing from the UI.
- Now includes: model selector, file uploader, document list with per-row
  delete buttons, and a "Refresh" control.
"""

import streamlit as st
from api_utils import delete_document, list_documents, upload_document


def display_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")

        # ------------------------------------------------------------------ #
        # Model selection
        # ------------------------------------------------------------------ #
        st.selectbox(
            "Model",
            options=["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            key="model",
            help="Select the Gemini model used to generate answers.",
        )

        st.divider()

        # ------------------------------------------------------------------ #
        # Document upload
        # ------------------------------------------------------------------ #
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "PDF, DOCX, or HTML",
            type=["pdf", "docx", "html"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            if st.button("Upload & Index", use_container_width=True):
                with st.spinner("Uploading…"):
                    result = upload_document(uploaded_file)
                if result:
                    st.success(f"Indexed as document #{result['file_id']}")
                    st.rerun()

        st.divider()

        # ------------------------------------------------------------------ #
        # Document library
        # ------------------------------------------------------------------ #
        st.subheader("📚 Document Library")

        col_refresh, _ = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        docs = list_documents()
        if not docs:
            st.caption("No documents indexed yet.")
        else:
            for doc in docs:
                col_name, col_btn = st.columns([3, 1])
                with col_name:
                    st.caption(f"**#{doc['id']}** {doc['filename']}")
                with col_btn:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Delete this document"):
                        with st.spinner("Deleting…"):
                            result = delete_document(doc["id"])
                        if result:
                            st.success("Deleted")
                            st.rerun()