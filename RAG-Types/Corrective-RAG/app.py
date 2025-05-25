import streamlit as st
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import tempfile
import os

# -------------------------------
# Helper Functions
# -------------------------------

@st.cache_resource
def setup_vectorstore(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name

    loader = TextLoader(tmpfile_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

    os.unlink(tmpfile_path)  # Clean up temp file
    return vectorstore

def generate_initial_answer(question, model="llama3"):
    response = ollama.generate(model=model, prompt=question)
    return response['response']

def needs_correction(initial_answer):
    low_confidence_phrases = ["i don't know", "not sure", "no information"]
    return any(phrase in initial_answer.lower() for phrase in low_confidence_phrases)

def retrieve_context(question, vectorstore, k=2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs])

def generate_corrected_answer(question, context, model="llama3"):
    full_prompt = f"""
    Based on the following context:
    {context}

    Answer the question:
    {question}
    """
    response = ollama.generate(model=model, prompt=full_prompt)
    return response['response']

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="🔍 Corrective RAG with Ollama", layout="centered")

st.title("🔍 Corrective RAG with Ollama")
st.markdown("Upload a document and ask questions — get accurate answers using local LLM + retrieval.")

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        st.success("Document uploaded successfully!")

    st.markdown("---")
    st.markdown("💡 Make sure Ollama is running locally.")
    st.markdown("🦙 Model used: `llama3` or similar")

if uploaded_file:
    with st.spinner("🧠 Setting up knowledge base..."):
        vectorstore = setup_vectorstore(uploaded_file)

    question = st.text_input("Ask a question about the document:")
    if st.button("🧮 Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🧠 Generating initial answer..."):
                initial_answer = generate_initial_answer(question)
                st.subheader("📝 Initial Answer")
                st.write(initial_answer)

                if needs_correction(initial_answer):
                    with st.spinner("🔍 Needs correction. Retrieving context..."):
                        context = retrieve_context(question, vectorstore)
                        st.subheader("📄 Retrieved Context")
                        st.write(context)

                        with st.spinner("🧮 Generating corrected answer..."):
                            corrected_answer = generate_corrected_answer(question, context)
                            st.subheader("✅ Final Corrected Answer")
                            st.write(corrected_answer)
                else:
                    st.success("✅ Confidence is high. No correction needed.")
else:
    st.info("Please upload a `.txt` document to begin.")
