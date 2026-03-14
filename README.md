# ReqPal

> AI-powered requirements management platform for regulatory compliance workflows.

## What This Does

ReqPal combines RAG (Retrieval-Augmented Generation) with a structured stakeholder and requirements workflow to help PMs turn compliance documents into validated user stories.

- **Document RAG** – Upload PDFs, DOCX, CSV, TXT, JSON and ask questions in natural language
- **Stakeholder Management** – Create stakeholders with dashboard access tokens
- **Requirements Gathering** – Stakeholders submit requirements via AI-guided chat
- **User Story Generation** – AI generates user stories grounded in your documents
- **PM Validation Workflow** – Stakeholder accept/reject → PM validates → Export to Jira
- **Export to Jira** – CSV and JSON export of PM-validated user stories

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Jo-llama/ReqPal.git
cd ReqPal
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API keys
```

### Running

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Open `http://localhost:8001` in your browser.

## Project Structure

```
ReqPal/
├── main.py                    # FastAPI application & all API endpoints
├── models.py                  # Pydantic data models
├── storage.py                 # SQLite persistence layer
├── db_schema.py               # SQLite schema definitions
├── requirements.txt           # Python dependencies
│
├── backend/
│   └── services/
│       ├── rag_service.py           # ChromaDB embedding & retrieval
│       ├── reranker_service.py      # Cross-encoder reranking
│       ├── llm_router.py            # Groq → OpenAI → Ollama fallback
│       ├── groq_http.py             # Groq API client
│       └── rag_llm_prompts.py       # Prompt templates
│
└── static/
    ├── index.html             # PM Dashboard
    └── dashboard.html         # Stakeholder Dashboard
```

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

LLM fallback chain: **Groq** → **OpenAI** → **Ollama** (local)

## Architecture

- **Backend**: FastAPI + SQLite (persistent storage) + ChromaDB (vector store)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, no API cost)
- **Reranker**: `BAAI/bge-reranker-base` (cross-encoder)
- **LLM**: Groq/OpenAI/Ollama via cascading fallback

## License

Proprietary
