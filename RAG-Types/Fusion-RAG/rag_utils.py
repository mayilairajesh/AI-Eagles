# rag_utils.py

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

DATA_PATH = "data"
DB_PATH = "db"

# Load and split docs
def load_docs():
    loader = DirectoryLoader(DATA_PATH, show_progress=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

# Create vector db
def create_db():
    documents = load_docs()
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embedding, persist_directory=DB_PATH)
    return vectordb

# Get retriever
def get_retriever():
    if not os.path.exists(DB_PATH):
        vectordb = create_db()
    else:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
    return vectordb.as_retriever(search_kwargs={"k": 5})

# Generate multiple queries
def generate_queries(query, llm):
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Given a question, generate {num} different ways to phrase it:\n\nQuestion: {question}\n\nAlternate queries:"
    ).format(num=3, question=query)
    response = llm.invoke(prompt)
    return [q.strip() for q in response.strip().split('\n') if q.strip()]

# Retrieve for each query
def retrieve_contexts(queries, retriever):
    all_docs = []
    for q in queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)
    return all_docs

# Final RAG chain
def run_fusion_rag(query):
    llm = Ollama(model="llama3")  # Make sure you have llama3 pulled via `ollama pull llama3`
    retriever = get_retriever()

    # Step 1: Query Generation
    queries = generate_queries(query, llm)

    # Step 2: Retrieve docs for all queries
    fused_docs = retrieve_contexts(queries, retriever)

    # Step 3: Deduplicate documents
    unique_docs = list({doc.page_content: doc for doc in fused_docs}.values())

    # Step 4: RAG Chain
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": lambda x: "\n\n".join([d.page_content for d in unique_docs]), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)
    return answer, unique_docs