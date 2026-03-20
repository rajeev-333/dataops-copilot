import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

def generate_report(question: str, sql_answer: str = "", rag_answer: str = "", dq_report: str = "") -> str:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Report Agent for a power grid DataOps system.
        Your job is to synthesize information from multiple sources into one clean, 
        professional final answer. Be concise but complete. 
        If some sections are empty, ignore them gracefully."""),
        ("human", """Original Question: {question}

SQL Database Answer:
{sql_answer}

Documentation Answer:
{rag_answer}

Data Quality Report:
{dq_report}

Please synthesize all available information into one clear, professional final answer.""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "question": question,
        "sql_answer": sql_answer or "No database query was needed.",
        "rag_answer": rag_answer or "No documentation lookup was needed.",
        "dq_report": dq_report or "No data quality check was performed."
    })
    return response.content

if __name__ == '__main__':
    report = generate_report(
        question="What is the anomaly situation across all sensors?",
        sql_answer="Sensor S002 has 12 anomalies, S001 has 3 anomalies.",
        rag_answer="Voltage above 255V is classified as ANOMALY. Emergency alert raised after 3 anomalies in 1 hour.",
        dq_report="• S002 voltage avg 267.8V exceeds 255V threshold\n• S005 is inactive but appearing in results"
    )
    print(report)
