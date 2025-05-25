import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch

# Load documents
@st.cache_resource
def load_docs(directory="doc"):
    docs = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, file))
            docs.extend(loader.load())
    return docs

# Create embeddings and vector store
@st.cache_resource
def setup_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(_docs, embeddings)
    return vectorstore

# Setup LLM
@st.cache_resource
def setup_llm():
    model_name = "gpt2"  # You can replace with "meta-llama/Llama-3-8b" or any HF model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        device=0 if torch.cuda.is_available() else -1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Setup QA chain
@st.cache_resource
def setup_qa_chain(_vectorstore, _llm):
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever()
    )

# Main app
def main():
    st.title("📄 RAG Chatbot with Streamlit")

    # Load data and prepare RAG components
    docs = load_docs()
    vectorstore = setup_vectorstore(docs)
    llm = setup_llm()
    qa_chain = setup_qa_chain(vectorstore, llm)

    # User input
    question = st.text_input("Ask a question about your documents:")

    if question:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": question})
            st.write("Answer:", response["result"])

if __name__ == "__main__":
    main()