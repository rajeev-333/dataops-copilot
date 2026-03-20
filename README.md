# 🤖 DataOps Copilot — Multi-Agent AI System

A production-grade DataOps copilot built with LangGraph, LangChain, and Groq LLM. Routes natural-language queries through specialized agents for power-grid data analytics.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.21-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3.21-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43.2-red)

## 🏗️ Architecture

```
User → Streamlit Chat UI → LangGraph Supervisor
                                    ├── 🗄️ SQL Agent (Groq Llama 3.3 70B)
                                    ├── 📄 RAG Agent (ChromaDB)
                                    ├── 🔍 DQ Agent 
                                    └── 📋 Report Agent
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

* Multi-Agent Routing — Supervisor automatically decides SQL vs RAG vs BOTH

* LLM SQL Generation — Converts natural language to executable SQL queries

* RAG Pipeline — ChromaDB + sentence-transformers over grid documentation

* Data Quality — Auto-detects anomalies, NULLs, inactive sensors

* Agent Trace — Click to see full reasoning chain + raw SQL results

* Rate Limit Handling — Auto-retry with exponential backoff

* Production UI — Streamlit chat with professional formatting
## 🛠️ Tech Stack
* 🤖 LLM: Groq (Llama 3.3 70B + Llama 3.1 8B)
* 🧠 Orchestration: LangGraph + LangChain 0.3.x
* 📊 Vector DB: ChromaDB 0.5.x (local)
* 🔍 Embeddings: sentence-transformers all-MiniLM-L6-v2
* 🗄️ Relational: SQLite (power grid sensor data)
* 🎨 UI: Streamlit 1.43.2
* ☁️ Deployed: Streamlit Cloud (Python 3.12)

## 🚀 Running Locally

### 1. Clone the repository

     git clone https://github.com/rajeev-333/dataops-copilot.git
    cd dataops-copilot

### 2. Create virtual environment

    python -m venv venv --without-pip
    venv\Scripts\activate        # Windows
    python -m ensurepip --upgrade
    pip install -r requirements.txt

### 3. Set up environment variables
 * Create a .env file:
      GROQ_API_KEY=your_groq_api_key_here
Get a free API key at console.groq.com

### 4. Initialize the database

    python utils/db_setup.py

### 5. Run the app
 
     streamlit run app/streamlit_app.py

## Project Structure
```
dataops-copilot/
├── agents/           # Individual agent implementations
│   ├── sql_agent.py
│   ├── rag_agent.py
│   ├── dq_agent.py
│   └── report_agent.py
├── app/              # Streamlit UI
│   └── streamlit_app.py
├── graph/            # LangGraph pipeline
│   └── pipeline.py
├── data/             # SQLite DB + docs
│   ├── sensor_data.db
│   └── docs/grid_manual.txt
├── vectorstore/      # ChromaDB (auto-created)
├── requirements.txt
├── README.md
└── .env             # Your API key (gitignored)
```

## Example Questions
* "How many anomalies were recorded in total?" → SQL Agent

* "What is the voltage threshold for an anomaly?" → RAG Agent

* "Summarize the anomaly situation and what the protocol says about it" → Both Agents

# 👤 Author
Rajeev Gupta — M.Tech Power Systems, NIT Warangal
AI/ML Engineer | Data Engineer | Power Systems Researcher
