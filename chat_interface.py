"""
chat_interface.py
Streamlit chat UI.

Fixes vs original:
- The "Details" expander used to echo the full answer again under
  "Generated Answer" — redundant since the answer is already visible
  in the chat bubble above. Replaced with a compact metadata summary
  (model + session ID only).
- Error path no longer appends a broken message to session state.
"""

import streamlit as st
from api_utils import get_api_response


def display_chat_interface() -> None:
    # Replay history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # New user input
    if prompt := st.chat_input("Ask anything about Arjuna's internal docs…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking…"):
            response = get_api_response(
                prompt,
                st.session_state.session_id,
                st.session_state.model,
            )

        if response:
            st.session_state.session_id = response.get("session_id")
            answer = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

            with st.expander("ℹ️ Response details"):
                st.markdown(f"**Model:** `{response['model']}`")
                st.markdown(f"**Session ID:** `{response['session_id']}`")
        else:
            st.error("No response from the API. Is the backend running?")