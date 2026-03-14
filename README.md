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
- Export accepted stories as CSV or JSON

### RAG Engine
- Local embeddings: `BAAI/bge-small-en-v1.5` (no API cost)
- Cross-encoder reranking: `BAAI/bge-reranker-base`
- Three ChromaDB collections: compliance documents, stakeholder requirements, user stories

### LLM — Qwen2.5 (local)
- Primary: **Qwen/Qwen2.5-7B-Instruct** running fully locally via HuggingFace transformers
- No external LLM API required
- Optional fallback: OpenAI (set `OPENAI_API_KEY`), Ollama (set `OLLAMA_MODEL`)
- Model loads once at startup, all inference stays on-device

### User Story Workflow
```
ai_generated → pending_review → accepted (stakeholder) → pm_validated → exported
```

### Export
- CSV (Jira-compatible: Summary, Description, Acceptance Criteria, Priority, Labels, Status, Issue Type)
- JSON (full story objects)
- Only PM-validated stories are exported

---

## Quick Start (Local)

### Prerequisites
- Python 3.9+
- GPU with 16 GB+ VRAM recommended (CPU also works, slower)

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
# Qwen model (default: Qwen/Qwen2.5-7B-Instruct)
QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct

# Set to 1 to load in 4-bit quantization (~8 GB VRAM instead of ~15 GB)
QWEN_LOAD_4BIT=0

# Optional fallback LLMs
# OPENAI_API_KEY=sk-...
# OLLAMA_MODEL=qwen2.5:7b
```

### Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Open **http://localhost:8001**

> First run downloads the Qwen model (~15 GB). Subsequent starts load from cache.

---

## Deploy on Lightning AI

### 1. Create a Studio

Go to [lightning.ai](https://lightning.ai), create a new **Studio** with a GPU instance (L4 or A10G recommended — 24 GB VRAM).

### 2. Clone and configure

```bash
git clone https://github.com/Jo-llama/ReqPal.git
cd ReqPal

# Create .env
cat > .env << 'ENV'
QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
QWEN_LOAD_4BIT=0
ENV
```

### 3. Start

```bash
bash startup.sh
```

The startup script installs dependencies and starts the server on port 8001.

### 4. Access

In Lightning AI Studio, use **Port Forwarding** to expose port `8001`. The PM Dashboard will be available at the forwarded URL.

### VRAM Guide

| Model | Float16 | 4-bit (`QWEN_LOAD_4BIT=1`) |
|---|---|---|
| Qwen2.5-7B-Instruct | ~15 GB | ~5 GB |
| Qwen2.5-14B-Instruct | ~30 GB | ~10 GB |
| Qwen2.5-72B-Instruct | ~150 GB | ~45 GB |

For Lightning AI free tier (T4, 16 GB), use `Qwen2.5-7B-Instruct` with `QWEN_LOAD_4BIT=0`, or `Qwen2.5-14B-Instruct` with `QWEN_LOAD_4BIT=1`.

---

## Project Structure

```
ReqPal/
├── main.py                          # FastAPI app — all API endpoints
├── models.py                        # Pydantic models & enums
├── storage.py                       # SQLite CRUD — all entities
├── db_schema.py                     # SQLite schema (13 tables)
├── requirements.txt
├── startup.sh                       # Lightning AI startup script
│
├── backend/
│   └── services/
│       ├── rag_service.py           # Chunking, embedding, ChromaDB retrieval
│       ├── reranker_service.py      # BGE cross-encoder reranking
│       ├── llm_router.py            # Qwen (local) → OpenAI → Ollama fallback
│       ├── groq_http.py             # Legacy HTTP client (unused)
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
| Persistence | SQLite |
| Vector Store | ChromaDB |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) |
| Reranker | `BAAI/bge-reranker-base` (local) |
| LLM | Qwen2.5-Instruct (local, HuggingFace transformers) |
| Frontend | Vanilla JS, no build step |

Everything runs locally — no mandatory external APIs.

---

## Supported Document Types

| Format | Processing |
|---|---|
| PDF | PyPDF2 |
| DOCX | python-docx |
| CSV | Row-by-row chunking |
| TXT | Plain text chunking |
| JSON | Structured data parsing |

---

## License

Proprietary
