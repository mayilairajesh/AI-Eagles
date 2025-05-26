import streamlit as st
import ollama
import tempfile
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Set page config — must be first
st.set_page_config(page_title="Self-RAG Chatbot", layout="centered")


# Load document and create vector store
def load_vectorstore(file_path, file_type):
    if file_type == "text/plain":
        loader = TextLoader(file_path)
    elif file_type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()


# Self-RAG prompt template
def self_rag_prompt(query, context):
    return f"""You are an AI assistant. Use the following pieces of context to answer the query.
If the context doesn't help, say so.

Context:
{context}

Query:
{query}

Answer:"""


# Generate response using RAG
def generate_response(query, retriever):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = self_rag_prompt(query, context)

    response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


# Streamlit UI
st.title("🧠 Self-RAG Chatbot with DeepSeek (via Ollama)")
st.subheader("Upload a document and ask questions about it!")

# File upload
uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
        tmpfile.write(uploaded_file.read())
        temp_file_path = tmpfile.name

    try:
        # Load and process file
        retriever = load_vectorstore(temp_file_path, uploaded_file.type)
        st.session_state["retriever"] = retriever
        st.success("Document loaded successfully!")
    finally:
        os.unlink(temp_file_path)  # Clean up temporary file

# Input box
user_input = st.text_input("Ask a question based on the uploaded document:")

if user_input and "retriever" in st.session_state:
    with st.spinner("Generating response..."):
        answer = generate_response(user_input, st.session_state["retriever"])
        st.markdown("### Answer:")
        st.write(answer)
elif user_input:
    st.warning("Please upload a document before asking questions.")