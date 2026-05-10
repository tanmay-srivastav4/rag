import streamlit as st

from frontend.api_utils import delete_document, list_documents, upload_document


def display_sidebar() -> None:
    with st.sidebar:
        st.header("Settings")

        st.selectbox(
            "Model",
            options=["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            key="model",
            help="Select the Gemini model used to generate answers.",
        )

        st.divider()
        st.subheader("Upload Document")

        uploaded_file = st.file_uploader(
            "PDF, DOCX, or HTML",
            type=["pdf", "docx", "html"],
            label_visibility="collapsed",
        )
        if uploaded_file and st.button("Upload and Index", use_container_width=True):
            with st.spinner("Uploading..."):
                result = upload_document(uploaded_file)
            if result:
                st.success(f"Indexed as document #{result['file_id']}")
                st.rerun()

        st.divider()
        st.subheader("Document Library")

        if st.button("Refresh", use_container_width=True):
            st.rerun()

        docs = list_documents()
        if not docs:
            st.caption("No documents indexed yet.")
            return

        for doc in docs:
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                st.caption(f"**#{doc['id']}** {doc['filename']}")
            with col_btn:
                if st.button("Delete", key=f"del_{doc['id']}", help="Delete this document"):
                    with st.spinner("Deleting..."):
                        result = delete_document(doc["id"])
                    if result:
                        st.success("Deleted")
                        st.rerun()