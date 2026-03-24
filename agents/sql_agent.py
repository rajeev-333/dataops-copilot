import os, sys, time, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.db'))

# Known correct table names — add any others your DB has
CORRECT_TABLES = ["sensor_readings", "sensors"]

def fix_sql(sql: str) -> str:
    """Auto-correct common LLM table name hallucinations."""
    corrections = {
        r'\bsensor_reading\b': 'sensor_readings',
        r'\bsensors_data\b': 'sensor_readings',
        r'\bsensor_data\b': 'sensor_readings',
        r'\breading\b': 'sensor_readings',
        r'\breadings\b': 'sensor_readings',
        r'\bsensor_table\b': 'sensor_readings',
    }
    for pattern, replacement in corrections.items():
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    return sql

def get_sql_answer(question: str) -> str:
    for attempt in range(3):
        try:
            db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
            schema = db.get_table_info()

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )

            # Step 1: Generate SQL
            sql_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert SQL generator for SQLite. Write a single valid SQLite query.
Return ONLY the raw SQL — no explanation, no markdown, no backticks.

CRITICAL: The database has EXACTLY these tables (do NOT invent other names):
- sensor_readings  (columns: id, sensor_id, location, timestamp, voltage, temperature, status)
- sensors          (columns: sensor_id, type, location, status)

Common aggregate examples:
- Average temperature → SELECT AVG(temperature) FROM sensor_readings
- Count anomalies    → SELECT COUNT(*) FROM sensor_readings WHERE status = 'ANOMALY'
- Max voltage        → SELECT MAX(voltage) FROM sensor_readings

Full schema:
{schema}"""),
                ("human", "{question}")
            ])

            sql_chain = sql_prompt | llm
            sql_response = sql_chain.invoke({"question": question})
            sql_query = sql_response.content.strip().strip("```sql").strip("```").strip()

            # Step 2: Auto-correct table name hallucinations
            sql_query_fixed = fix_sql(sql_query)
            if sql_query_fixed != sql_query:
                print(f"🔧 SQL auto-corrected: {sql_query} → {sql_query_fixed}")
            sql_query = sql_query_fixed

            print(f"\n📝 Generated SQL: {sql_query}\n")

            # Step 3: Execute
            try:
                result = db.run(sql_query)
            except Exception as sql_err:
                return f"SQL_FAILED: {str(sql_err)[:100]}"

            if not result or result == "[]":
                return "SQL_FAILED: query returned no results"

            # Step 4: Format answer
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data analyst. Answer the question clearly using the SQL result. Be concise and factual."),
                ("human", "Question: {question}\nSQL: {sql}\nResult: {result}\n\nClear answer:")
            ])
            answer_chain = answer_prompt | llm
            final = answer_chain.invoke({
                "question": question,
                "sql": sql_query,
                "result": result
            })
            return final.content

        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 20
                print(f"⏳ SQL rate limit. Waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                return f"SQL_FAILED: {err[:150]}"
    return "SQL_FAILED: rate limit after 3 retries"
