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

def run_dq_check(sql_result: str) -> str:
    if not sql_result or "SQL_FAILED" in sql_result or sql_result.strip() == "":
        return "⚠️ DQ check skipped — no SQL result available."
    for attempt in range(3):
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Data Quality Agent.
Analyze ONLY the actual SQL result below. Do NOT invent numbers.
Check for: NULL values, anomalous readings, inactive sensors.
Be concise — max 4 bullet points. If data looks clean, say so."""),
                ("human", "SQL Result:\n{sql_result}\n\nBrief DQ report:")
            ])
            chain = prompt | llm
            response = chain.invoke({"sql_result": sql_result})
            return response.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 15
                print(f"⏳ DQ rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"DQ check skipped: {err[:80]}"
    return "DQ check skipped: rate limit."
