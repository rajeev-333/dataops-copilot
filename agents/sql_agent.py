import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.db'))

FAIL_PHRASES = [
    "i cannot", "unable to determine", "iteration limit", "time limit",
    "not provided", "hypothetical", "i don't have", "i'm unable",
    "cannot determine", "no information"
]

def get_sql_answer(question: str) -> str:
    for attempt in range(3):
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
            agent = create_sql_agent(
                llm=llm,
                db=db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15,
                max_execution_time=60,
                prefix="""You are an expert SQL agent. You MUST always:
1. First call sql_db_list_tables to see available tables
2. Then call sql_db_schema on the relevant table
3. Then write and execute a SQL query
4. Return the actual result

Never repeat the same action twice. Move forward step by step."""
            )
            result = agent.invoke({"input": question})
            answer = result.get("output", "")
            if any(p in answer.lower() for p in FAIL_PHRASES):
                return "SQL_FAILED"
            return answer
        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 20
                print(f"⏳ SQL rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"SQL_FAILED: {err[:100]}"
    return "SQL_FAILED: rate limit after 3 retries"
