import streamlit as st
import ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Set page config — must come FIRST
st.set_page_config(page_title="Self-RAG Chatbot", layout="centered")


# Load document and create vector store
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("data.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()


retriever = load_vectorstore()


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
def generate_response(query):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = self_rag_prompt(query, context)

    response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


# UI
st.title("🧠 Self-RAG Chatbot with DeepSeek (via Ollama)")

# Input box
user_input = st.text_input("Ask a question based on your knowledge base:")

if user_input:
    with st.spinner("Generating response..."):
        answer = generate_response(user_input)
        st.markdown("### Answer:")
        st.write(answer)