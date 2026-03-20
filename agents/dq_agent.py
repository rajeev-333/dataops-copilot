import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

def get_dq_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

def check_data_quality(sql_result: str) -> str:
    llm = get_dq_agent()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Data Quality Agent for a power grid monitoring system.
        Analyze the given SQL query result and check for:
        1. Missing or NULL values
        2. Anomalous readings (voltage > 255V or < 200V, temperature > 60C, frequency outside 49.5-50.5 Hz)
        3. Inactive sensors in the results
        4. Data completeness issues
        Provide a concise data quality report in 3-4 bullet points."""),
        ("human", "Analyze this query result for data quality issues:\n\n{result}")
    ])
    chain = prompt | llm
    response = chain.invoke({"result": sql_result})
    return response.content

if __name__ == '__main__':
    sample = """
    sensor_id | location     | avg_voltage | anomaly_count
    S001      | Substation_A | 231.5       | 3
    S002      | Substation_B | 267.8       | 12
    S005      | Substation_E | 229.1       | 0
    """
    print(check_data_quality(sample))
