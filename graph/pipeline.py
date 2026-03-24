import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

class GraphState(TypedDict):
    question: str
    intent: str
    sql_answer: Optional[str]
    rag_answer: Optional[str]
    dq_report: Optional[str]
    final_answer: Optional[str]

def supervisor_node(state: GraphState) -> GraphState:
    for attempt in range(3):
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Classify the question into ONE of: sql, rag, both

sql → needs live database data (counts, values, records, highest, lowest, average)
rag → needs documentation (thresholds, protocols, procedures, rules, why a sensor is inactive)
both → needs both database AND documentation

Examples:
"How many anomalies?" → sql
"What is voltage threshold?" → rag
"Which sensor is inactive and why?" → rag
"Summarize anomaly situation and protocol" → both

Reply with ONLY one word: sql, rag, or both"""),
                ("human", "{question}")
            ])
            chain = prompt | llm
            response = chain.invoke({"question": state["question"]})
            intent = response.content.strip().lower().strip(".")
            if intent not in ["sql", "rag", "both"]:
                intent = "both"
            print(f"\n🧠 Supervisor → [{intent.upper()}]\n")
            time.sleep(0.3)
            return {**state, "intent": intent}
        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 15
                print(f"⏳ Supervisor rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                return {**state, "intent": "both"}
    return {**state, "intent": "both"}

def sql_node(state: GraphState) -> GraphState:
    print("\n🗄️  SQL Agent querying...\n")
    from agents.sql_agent import get_sql_answer
    result = get_sql_answer(state["question"])
    if "SQL_FAILED" in result:
        print("⚠️ SQL failed")
        return {**state, "sql_answer": None}
    return {**state, "sql_answer": result}

def rag_node(state: GraphState) -> GraphState:
    print("\n📄 RAG Agent searching docs...\n")
    from agents.rag_agent import get_rag_answer
    result = get_rag_answer(state["question"])
    return {**state, "rag_answer": result}

def dq_node(state: GraphState) -> GraphState:
    print("\n🔍 DQ Agent checking...\n")
    from agents.dq_agent import run_dq_check
    result = run_dq_check(state.get("sql_answer", ""))
    return {**state, "dq_report": result}

def report_node(state: GraphState) -> GraphState:
    print("\n📋 Report Agent writing...\n")
    from agents.report_agent import generate_report
    result = generate_report(
        question=state["question"],
        sql_answer=state.get("sql_answer"),
        rag_answer=state.get("rag_answer"),
        dq_report=state.get("dq_report")
    )
    return {**state, "final_answer": result}

def route_after_supervisor(state: GraphState) -> str:
    return state["intent"]

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql", sql_node)
    graph.add_node("rag", rag_node)
    graph.add_node("dq", dq_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "sql": "sql",
        "rag": "rag",
        "both": "sql"
    })
    graph.add_edge("sql", "dq")
    graph.add_edge("dq", "report")
    graph.add_edge("rag", "report")
    graph.add_edge("report", END)

    # For "both": after sql→dq→report but also need rag
    # Override: both routes to sql first, then rag runs in report
    return graph.compile()
