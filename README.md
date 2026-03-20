# 🤖 DataOps Copilot — Multi-Agent AI System

A production-grade AI-first application built with LangGraph, LangChain, and Groq LLM.
Demonstrates a multi-agent architecture for power grid data operations.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.21-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3.21-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43.2-red)

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI → LangGraph Supervisor
├── SQL Agent (natural language → SQLite)
├── RAG Agent (semantic search → ChromaDB)
├── DQ Agent (data quality analysis)
└── Report Agent (final synthesis)
```


## 🤖 Agent Descriptions

| Agent | Model | Role |
|-------|-------|------|
| 🧠 Supervisor | Llama 3.1 8B | Classifies intent → routes to SQL / RAG / BOTH |
| 🗄️ SQL Agent | Llama 3.3 70B | Generates & executes SQL from natural language |
| 📄 RAG Agent | Llama 3.1 8B | Retrieves answers from indexed documentation |
| 🔍 DQ Agent | Llama 3.1 8B | Flags data quality issues in query results |
| 📋 Report Agent | Llama 3.1 8B | Synthesizes all agent outputs into final answer |

## ✨ Features

- **Conversational AI** — multi-turn chat with session memory
- **Intelligent Routing** — Supervisor automatically decides which agents to call
- **RAG Pipeline** — documents indexed in ChromaDB with sentence-transformers embeddings
- **Agentic SQL** — LLM reasons step-by-step to write and execute correct SQL
- **Data Quality** — every SQL result is automatically analyzed for anomalies
- **Agent Trace** — expandable panel showing exactly which agents ran and what they found
- **Rate Limit Handling** — auto-retry with exponential backoff on API limits

## 🛠️ Tech Stack

- **LLM**: Groq API (Llama 3.3 70B + Llama 3.1 8B)
- **Agent Orchestration**: LangGraph + LangChain
- **Vector Database**: ChromaDB (local)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Relational DB**: SQLite
- **UI**: Streamlit
- **Environment**: Python 3.12, dotenv

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/rajeev-333/dataops-copilot.git
cd dataops-copilot

### 2. Create virtual environment
```bash
python -m venv venv --without-pip
venv\Scripts\activate        # Windows
python -m ensurepip --upgrade
pip install -r requirements.txt

### 3. Set up environment variables
 * Create a .env file:
      GROQ_API_KEY=your_groq_api_key_here
Get a free API key at console.groq.com

### 4. Initialize the database
```bash 
python utils/db_setup.py

### 5. Run the app
```bash 
streamlit run app/streamlit_app.py


## Example Questions
* "How many anomalies were recorded in total?" → SQL Agent

* "What is the voltage threshold for an anomaly?" → RAG Agent

* "Summarize the anomaly situation and what the protocol says about it" → Both Agents

# 👤 Author
Rajeev Gupta — M.Tech Power Systems, NIT Warangal
AI/ML Engineer | Data Engineer | Power Systems Researcher