import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import os

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

# Setup LLM with Ollama
@st.cache_resource
def setup_llm():
    return OllamaLLM(model="llama3")

# Prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If the context does not provide enough information, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Setup QA chain
@st.cache_resource
def setup_qa_chain(_vectorstore, _llm):
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

# Main UI
def main():
    st.title("📄 RAG Chatbot with Ollama")

    docs = load_docs()
    vectorstore = setup_vectorstore(docs)
    llm = setup_llm()
    qa_chain = setup_qa_chain(vectorstore, llm)

    question = st.text_input("Ask a question about your documents:")

    if question:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": question})
            st.write("Answer:", result["result"])

if __name__ == "__main__":
    main()