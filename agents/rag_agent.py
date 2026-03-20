import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

DOCS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'docs')
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectorstore')

def _is_vectorstore_populated():
    chroma_db = os.path.join(VECTORSTORE_PATH, 'chroma.sqlite3')
    return os.path.exists(chroma_db)

def build_vectorstore():
    documents = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith('.txt'):
            loader = TextLoader(os.path.join(DOCS_PATH, filename), encoding='utf-8')
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    print(f"✅ Vectorstore built with {len(chunks)} chunks from {len(documents)} documents")
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def get_rag_chain(rebuild=False):
    if rebuild or not _is_vectorstore_populated():
        vectorstore = build_vectorstore()
    else:
        vectorstore = load_vectorstore()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
    return chain

def ask(question: str, rebuild=False) -> str:
    chain = get_rag_chain(rebuild=rebuild)
    result = chain.invoke({"query": question})
    return result["result"]

if __name__ == '__main__':
    print("\n📄 DataOps RAG Agent — Ask questions from the Grid Manual\n")
    print("(Type 'rebuild' to re-index documents, 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        if question.lower() == 'rebuild':
            get_rag_chain(rebuild=True)
            print("✅ Vectorstore rebuilt!\n")
            continue
        if not question:
            continue
        answer = ask(question)
        print(f"\nAgent: {answer}\n")
