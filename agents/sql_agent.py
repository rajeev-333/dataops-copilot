import os
import sys
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

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.db')

def get_sql_answer(question: str) -> str:
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        db = SQLDatabase.from_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=60
        )
        result = agent.invoke({"input": question})
        answer = result.get("output", "")
        # Reject hallucinated fallback answers
        if any(phrase in answer.lower() for phrase in [
            "i cannot", "i'm unable", "unable to determine",
            "iteration limit", "time limit", "not provided",
            "hypothetical", "i don't have"
        ]):
            return "SQL_FAILED"
        return answer
    except Exception as e:
        return f"SQL_FAILED: {str(e)[:100]}"

if __name__ == '__main__':
    while True:
        q = input("You: ").strip()
        if q.lower() in ['exit', 'quit']:
            break
        print("Agent:", get_sql_answer(q))
