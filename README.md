# ReqPal

**AI-powered requirements management platform for regulatory compliance workflows.**

ReqPal helps Product Managers turn compliance documents and stakeholder input into validated, Jira-ready user stories — using RAG, LLM-guided chat, and a structured review workflow.

---

## How It Works

```
Documents → RAG → Stakeholder Chat → Requirements → AI User Stories → Review → Export
```

1. **PM uploads compliance documents** (PDFs, DOCX, regulations, specs)
2. **PM creates stakeholders** and shares their personal dashboard link
3. **Stakeholders chat with an AI assistant** that extracts structured requirements from the conversation
4. **AI generates user stories** grounded in both stakeholder requirements and uploaded documents
5. **Stakeholders review** generated stories — accept or reject each one
6. **PM validates** accepted stories and exports to Jira (CSV or JSON)

---

## Features

### PM Dashboard (`/`)
- **Ask tab** — RAG search and Q&A over uploaded documents (Answer mode + raw chunk Search mode)
- **Stakeholder Requirements tab** — view and manage all requirements submitted via stakeholder dashboards
- **User Stories tab** — full CRUD, group/filter by priority/category/status, PM validation, Jira export

### Stakeholder Dashboard (`/dashboard/{token}`)
- Token-based access, no login required
- AI Requirements Assistant — guided chat that extracts and saves structured requirements
- Review generated user stories — accept or reject with a single click
- View accepted stories and export them as CSV or JSON

### RAG Engine
- Local embeddings: `BAAI/bge-small-en-v1.5` (no API cost)
- Cross-encoder reranking: `BAAI/bge-reranker-base`
- Three ChromaDB collections: compliance documents, stakeholder requirements, user stories
- LLM fallback chain: **Groq → OpenAI → Ollama**

### User Story Workflow
```
ai_generated → pending_review → accepted (stakeholder) → pm_validated → exported
```

### Export
- CSV (Jira-compatible: Summary, Description, Acceptance Criteria, Priority, Labels, Status, Issue Type)
- JSON (full story objects)
- Only PM-validated stories are exported

---

## Quick Start

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com) (free tier works) or OpenAI key

### Installation

```bash
git clone https://github.com/Jo-llama/ReqPal.git
cd ReqPal
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # optional fallback
```

### Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Open **http://localhost:8001** — the PM Dashboard loads immediately.

> First run downloads the embedding and reranker models (~150 MB total). Subsequent starts are instant.

---

## Project Structure

```
ReqPal/
├── main.py                          # FastAPI app — all API endpoints
├── models.py                        # Pydantic models & enums
├── storage.py                       # SQLite CRUD — all entities
├── db_schema.py                     # SQLite schema (13 tables)
├── requirements.txt
│
├── backend/
│   └── services/
│       ├── rag_service.py           # Chunking, embedding, ChromaDB retrieval
│       ├── reranker_service.py      # BGE cross-encoder reranking
│       ├── llm_router.py            # Groq → OpenAI → Ollama fallback
│       ├── groq_http.py             # Groq HTTP client
│       └── rag_llm_prompts.py       # All LLM prompt templates
│
└── static/
    ├── index.html                   # PM Dashboard (tabbed)
    └── dashboard.html               # Stakeholder Dashboard
```

---

## Architecture

| Layer | Technology |
|---|---|
| API | FastAPI |
| Persistence | SQLite (via `storage.py`) |
| Vector Store | ChromaDB |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) |
| Reranker | `BAAI/bge-reranker-base` (local) |
| LLM | Groq / OpenAI / Ollama (cascading) |
| Frontend | Vanilla JS, no build step |

Storage is fully local — no external database required. ChromaDB and SQLite files are created automatically under `storage/` on first run.

---

## Supported Document Types

| Format | Processing |
|---|---|
| PDF | Text extraction via PyPDF2 |
| DOCX | Paragraph extraction via python-docx |
| CSV | Row-by-row chunking |
| TXT | Plain text chunking |
| JSON | Structured data parsing |

---

## License

Proprietary
