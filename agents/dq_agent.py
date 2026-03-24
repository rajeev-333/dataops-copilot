import os
import sys
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
    # Don't hallucinate if SQL failed
    if not sql_result or "SQL_FAILED" in sql_result or sql_result.strip() == "":
        return "⚠️ Data quality check skipped — SQL query did not return results."

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Data Quality Agent. Analyze ONLY the actual SQL query result provided.
            Do NOT invent or estimate numbers. Only report what is explicitly visible in the data.
            Check for: NULL or missing values, anomalous readings, inactive sensors.
            If the result is clean, say so. Be concise — max 5 bullet points."""),
            ("human", "SQL Result:\n{sql_result}\n\nProvide a brief data quality report based ONLY on the above data.")
        ])
        chain = prompt | llm
        response = chain.invoke({"sql_result": sql_result})
        return response.content
    except Exception as e:
        return f"DQ check error: {str(e)[:100]}"
