import streamlit as st

def display_sidebar():
    model_options = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")