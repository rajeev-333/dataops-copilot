import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

DOCS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'docs', 'grid_manual.txt'))

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    print("📚 Building vectorstore from docs...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # In-memory only — works on any filesystem
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print(f"✅ Vectorstore ready with {len(chunks)} chunks")
    return _vectorstore

def get_rag_answer(question: str) -> str:
    for attempt in range(3):
        try:
            vectorstore = get_vectorstore()
            docs = vectorstore.similarity_search(question, k=4)
            if not docs:
                return "RAG_FAILED: no relevant docs found"
            context = "\n\n".join([d.page_content for d in docs])

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a documentation expert for power grid systems.
Answer ONLY using the provided context. Be specific and concise.
If the answer is not in the context, say: 'This information is not in the documentation.'"""),
                ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            ])
            chain = prompt | llm
            response = chain.invoke({"context": context, "question": question})
            return response.content

        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 15
                print(f"⏳ RAG rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"RAG_FAILED: {err[:100]}"
    return "RAG_FAILED: rate limit after 3 retries"
