# app.py

import streamlit as st
from rag_utils import run_fusion_rag

st.set_page_config(page_title="Fusion RAG with Ollama", layout="centered")

st.title("🧠 Fusion RAG Chatbot (Local Ollama + Streamlit)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask something about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from Fusion RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, docs = run_fusion_rag(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Optional: Show sources
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(docs[:3]):
                    st.markdown(f"**Source {i + 1}:**\n{doc.page_content[:300]}...")