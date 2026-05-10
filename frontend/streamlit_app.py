import streamlit as st

from frontend.chat_interface import display_chat_interface
from frontend.sidebar import display_sidebar


def main() -> None:
    st.set_page_config(
        page_title="Arjuna Knowledge Assistant",
        page_icon="A",
        layout="wide",
    )

    st.title("Arjuna Knowledge Assistant")
    st.caption("Ask questions about internal company documents. Sensitive data is protected.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "model" not in st.session_state:
        st.session_state.model = "gemini-2.5-flash"

    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()