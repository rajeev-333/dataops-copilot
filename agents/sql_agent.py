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

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.db')

def get_sql_agent():
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
        max_execution_time=90
    )
    return agent

def ask(question: str) -> str:
    agent = get_sql_agent()
    try:
        result = agent.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"SQL query could not complete: {str(e)[:200]}"

if __name__ == '__main__':
    print("\n🤖 DataOps SQL Agent — Type your question (or 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        if not question:
            continue
        print("\nAgent: ", end="")
        answer = ask(question)
        print(answer)
        print()
