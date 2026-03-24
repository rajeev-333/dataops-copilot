import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

DOCS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'docs', 'grid_manual.txt'))

def _build_vectorstore():
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print("📚 Building FAISS vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    print(f"✅ FAISS vectorstore ready: {len(chunks)} chunks")
    return vs

try:
    import streamlit as st
    @st.cache_resource
    def get_vectorstore():
        return _build_vectorstore()
except Exception:
    _vs_cache = None
    def get_vectorstore():
        global _vs_cache
        if _vs_cache is None:
            _vs_cache = _build_vectorstore()
        return _vs_cache

def get_rag_answer(question: str) -> str:
    for attempt in range(3):
        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate

            vectorstore = get_vectorstore()
            docs = vectorstore.similarity_search(question, k=4)
            if not docs:
                return "This information is not in the documentation."

            context = "\n\n".join([d.page_content for d in docs])
            print(f"📄 RAG context retrieved ({len(docs)} chunks)")

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
            print(f"❌ RAG error (attempt {attempt+1}): {err[:150]}")
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 15
                print(f"⏳ Waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"RAG could not retrieve answer: {err[:100]}"
    return "RAG_FAILED: rate limit after 3 retries"
