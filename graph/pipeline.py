import os
import sys

# Fix path BEFORE all imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import time

# Import all agents at top level
from agents.sql_agent import ask as sql_ask
from agents.rag_agent import ask as rag_ask
from agents.dq_agent import check_data_quality
from agents.report_agent import generate_report

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── State shared across all agents ──────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    intent: str
    sql_answer: Optional[str]
    rag_answer: Optional[str]
    dq_report: Optional[str]
    final_answer: Optional[str]

# ── Node 1: Supervisor ───────────────────────────────────────────────────────
def supervisor_node(state: GraphState) -> GraphState:
    from groq import RateLimitError
    import time

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Supervisor Agent. Classify the user question into one of these intents:
        - 'sql': question needs data from the database (counts, averages, specific readings, anomaly records)
        - 'rag': question needs information from documentation (thresholds, protocols, procedures, definitions)
        - 'both': question needs both database data AND documentation

        Reply with ONLY one word: sql, rag, or both"""),
        ("human", "{question}")
    ])
    chain = prompt | llm

    for attempt in range(3):
        try:
            response = chain.invoke({"question": state["question"]})
            intent = response.content.strip().lower()
            if intent not in ["sql", "rag", "both"]:
                intent = "both"
            print(f"\n🧠 Supervisor Decision: route to [{intent.upper()}]\n")
            time.sleep(1)
            return {**state, "intent": intent}
        except RateLimitError:
            wait = (attempt + 1) * 20
            print(f"⏳ Rate limit hit. Waiting {wait}s before retry {attempt+1}/3...")
            time.sleep(wait)

    print("⚠️ Rate limit exceeded after 3 retries. Defaulting to 'both'.")
    return {**state, "intent": "both"}


# ── Node 2: SQL Agent ────────────────────────────────────────────────────────
def sql_node(state: GraphState) -> GraphState:
    print("🗄️  SQL Agent: querying database...")
    answer = sql_ask(state["question"])
    return {**state, "sql_answer": answer}

# ── Node 3: RAG Agent ────────────────────────────────────────────────────────
def rag_node(state: GraphState) -> GraphState:
    print("📄 RAG Agent: searching documentation...")
    answer = rag_ask(state["question"])
    return {**state, "rag_answer": answer}

# ── Node 4: Data Quality Agent ───────────────────────────────────────────────
def dq_node(state: GraphState) -> GraphState:
    if not state.get("sql_answer"):
        return state
    print("🔍 DQ Agent: checking data quality...")
    report = check_data_quality(state["sql_answer"])
    return {**state, "dq_report": report}

# ── Node 5: Report Agent ─────────────────────────────────────────────────────
def report_node(state: GraphState) -> GraphState:
    print("📋 Report Agent: generating final answer...")
    final = generate_report(
        question=state["question"],
        sql_answer=state.get("sql_answer", ""),
        rag_answer=state.get("rag_answer", ""),
        dq_report=state.get("dq_report", "")
    )
    return {**state, "final_answer": final}

# ── Routing Logic ────────────────────────────────────────────────────────────
def route_after_supervisor(state: GraphState) -> str:
    return state["intent"]

def route_after_sql(state: GraphState) -> str:
    return state["intent"]

# ── Build the Graph ──────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("dq_agent", dq_node)
    graph.add_node("report_agent", report_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "sql": "sql_agent",
        "rag": "rag_agent",
        "both": "sql_agent"
    })

    graph.add_conditional_edges("sql_agent", route_after_sql, {
        "sql": "dq_agent",
        "both": "rag_agent"
    })

    graph.add_edge("rag_agent", "dq_agent")
    graph.add_edge("dq_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()

# ── Main Chat Loop ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🤖 DataOps Copilot — Multi-Agent System")
    print("=" * 50)
    print("Ask anything about your power grid data or documentation.")
    print("Type 'exit' to quit.\n")

    app = build_graph()

    while True:
        question = input("You: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        if not question:
            continue

        print()
        result = app.invoke({
            "question": question,
            "intent": "",
            "sql_answer": None,
            "rag_answer": None,
            "dq_report": None,
            "final_answer": None
        })

        print("\n" + "=" * 50)
        print("💡 FINAL ANSWER:")
        print("=" * 50)
        print(result["final_answer"])
        print("=" * 50 + "\n")
