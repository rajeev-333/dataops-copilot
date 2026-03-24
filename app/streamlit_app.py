import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from graph.pipeline import build_graph

st.set_page_config(page_title="DataOps Copilot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #1a1a2e, #16213e);
        padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
    .agent-badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold; margin: 2px; }
    .badge-sql { background-color: #1e40af; color: white; }
    .badge-rag { background-color: #065f46; color: white; }
    .badge-both { background-color: #7c2d12; color: white; }
    .trace-box { background-color: #1e293b; border-radius: 8px;
        padding: 12px; font-size: 13px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='main-header'>
    <h1 style='color: white; margin: 0;'>🤖 DataOps Copilot</h1>
    <p style='color: #94a3b8; margin: 5px 0 0 0;'>
        Multi-Agent AI System — Power Grid Monitoring
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 Agent Architecture")
    st.markdown("""
| Agent | Role |
|-------|------|
| 🧠 Supervisor | Routes your question |
| 🗄️ SQL Agent | Queries the database |
| 📄 RAG Agent | Searches documentation |
| 🔍 DQ Agent | Checks data quality |
| 📋 Report Agent | Writes final answer |
    """)
    st.markdown("---")
    st.markdown("## 💡 Try These Questions")
    sample_questions = [
        "How many anomalies were recorded in total?",
        "Which sensor had the highest voltage?",
        "What is the voltage threshold for an anomaly?",
        "Which sensor is inactive and why?",
        "Summarize the anomaly situation and protocol",
        "What is the average temperature across all sensors?",
        "How many sensors are currently active?",
        "Which location has the most anomalies?"
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm your **DataOps Copilot** for Power Grid Monitoring.\n\nI can answer questions about:\n- 📊 **Live sensor data** (anomalies, voltages, temperatures)\n- 📄 **Grid documentation** (thresholds, protocols, procedures)\n- 🔍 **Data quality** issues\n\nTry asking: *'Summarize the anomaly situation and what the protocol says about it'*",
        "intent": None,
        "trace": None
    })

@st.cache_resource
def load_graph():
    return build_graph()

if "graph" not in st.session_state:
    with st.spinner("🔧 Initializing agents..."):
        st.session_state.graph = load_graph()


def run_agents(question):
    try:
        result = st.session_state.graph.invoke({
            "question": question,
            "intent": "",
            "sql_answer": None,
            "rag_answer": None,
            "dq_report": None,
            "final_answer": None
        })
        return result, None
    except Exception as e:
        return None, str(e)

def build_trace(result):
    trace = []
    intent = result.get("intent", "")
    trace.append(f"🧠 **Supervisor** → Routed to `{intent.upper()}`")
    if result.get("sql_answer"):
        trace.append(f"🗄️ **SQL Agent** → Query executed successfully")
        trace.append(f"🔍 **DQ Agent** → Data quality check completed")
    if result.get("rag_answer"):
        trace.append(f"📄 **RAG Agent** → Documentation retrieved")
    trace.append(f"📋 **Report Agent** → Final answer generated")
    return trace

def display_message(msg):
    with st.chat_message(msg["role"]):
        if msg.get("intent"):
            badge_class = f"badge-{msg['intent']}"
            st.markdown(
                f"<span class='agent-badge {badge_class}'>🧠 Routed to: {msg['intent'].upper()}</span>",
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 View Agent Trace", expanded=False):
                for step in msg["trace"]:
                    st.markdown(f"→ {step}")
                if msg.get("sql_answer"):
                    st.markdown("**📊 Raw SQL Result:**")
                    st.code(msg["sql_answer"], language="text")
                if msg.get("dq_report"):
                    st.markdown("**🔍 DQ Report:**")
                    st.markdown(msg["dq_report"])

for msg in st.session_state.messages:
    display_message(msg)

def handle_question(question):
    st.session_state.messages.append({"role": "user", "content": question,
                                       "intent": None, "trace": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents working..."):
            result, error = run_agents(question)

        if error:
            msg_content = f"⚠️ **Error:** {error[:200]}\n\nPlease try again."
            st.error(msg_content)
            st.session_state.messages.append({
                "role": "assistant", "content": msg_content,
                "intent": None, "trace": None
            })
        else:
            intent = result.get("intent", "")
            trace = build_trace(result)
            final_answer = result["final_answer"]

            st.markdown(
                f"<span class='agent-badge badge-{intent}'>🧠 Routed to: {intent.upper()}</span>",
                unsafe_allow_html=True
            )
            st.markdown(final_answer)

            with st.expander("🔍 View Agent Trace", expanded=False):
                for step in trace:
                    st.markdown(f"→ {step}")
                if result.get("sql_answer"):
                    st.markdown("**📊 Raw SQL Result:**")
                    st.code(result["sql_answer"], language="text")
                if result.get("dq_report"):
                    st.markdown("**🔍 DQ Report:**")
                    st.markdown(result["dq_report"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "intent": intent,
                "trace": trace,
                "sql_answer": result.get("sql_answer", ""),
                "dq_report": result.get("dq_report", "")
            })

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    handle_question(question)
    st.rerun()

if question := st.chat_input("Ask anything about your power grid data or documentation..."):
    handle_question(question)
