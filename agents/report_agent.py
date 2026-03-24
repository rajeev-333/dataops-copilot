import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

def generate_report(question: str, sql_answer: str = None,
                    rag_answer: str = None, dq_report: str = None) -> str:
    parts = []
    if sql_answer and "SQL_FAILED" not in sql_answer:
        parts.append(f"Database Result:\n{sql_answer}")
    if rag_answer and "RAG_FAILED" not in rag_answer:
        parts.append(f"Documentation:\n{rag_answer}")
    if dq_report and "skipped" not in dq_report:
        parts.append(f"Data Quality:\n{dq_report}")

    if not parts:
        return "⚠️ Could not retrieve an answer. Please wait a moment and try again."

    context = "\n\n---\n\n".join(parts)

    for attempt in range(3):
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Report Agent. Write a clear, professional answer.
Use ONLY the data provided. Do NOT invent numbers or facts.
Use markdown formatting. Be concise."""),
                ("human", f"Question: {question}\n\nData:\n{context}\n\nFinal Answer:")
            ])
            chain = prompt | llm
            response = chain.invoke({})
            return response.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 15
                print(f"⏳ Report rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"Report error: {err[:100]}"
    return "⚠️ Rate limit reached. Please try again in 1 minute."
