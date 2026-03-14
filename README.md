# ReqPal

**Your requirements companion — AI-powered requirements management for any domain.**

ReqPal helps Product Managers turn compliance documents and stakeholder input into validated, Jira-ready user stories using RAG, LLM-guided chat, and a structured review workflow.

---

## How It Works

```
Documents → RAG → Stakeholder Chat → Requirements → AI User Stories → Review → Export
```

1. **PM uploads documents** (PDFs, DOCX, regulations, specs, CSV, JSON)
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
- Local embeddings: `BAAI/bge-small-en-v1.5` (CPU, no API cost)
- Cross-encoder reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, 22 MB)
- Three ChromaDB collections: compliance documents, stakeholder requirements, user stories
- Query rewriting + domain-aware terminology enrichment

### LLM Providers (cascading fallback)

| Priority | Provider | Config |
|---|---|---|
| 1 | **Groq** (primary) | `GROQ_API_KEY` — fast, free tier, `llama-3.3-70b-versatile` |
| 2 | **Lightning AI** | `LIGHTNING_API_KEY` — fallback on Groq rate limit |
| 3 | **LitServe** (local Qwen) | `LITSERVE_ENABLED=1` — opt-in, requires GPU |
| 4 | **Qwen local** (transformers) | `QWEN_LOCAL_ENABLED=1` — opt-in, requires GPU |

The app works **without a GPU** when Groq or another cloud provider is configured.

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
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (recommended — no GPU needed)

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
# Primary LLM — get a free key at console.groq.com
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile   # optional, this is the default

# Optional cloud fallback (Lightning AI)
# LIGHTNING_API_KEY=your_lightning_key
# LIGHTNING_MODEL=lightning-ai/llama-3.3-70b

# Optional local fallbacks (require GPU)
# LITSERVE_ENABLED=1
# QWEN_LOCAL_ENABLED=1
# QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct

# Optional additional fallbacks
# OPENAI_API_KEY=sk-...
# OLLAMA_MODEL=qwen2.5:7b
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

Open **http://localhost:8001**

---

## Deploy on Railway (Docker)

The recommended production deployment — CPU-only, no GPU required.

### 1. Connect your repo

Go to [railway.app](https://railway.app), create a new project, and connect your GitHub repo. Railway auto-detects the `Dockerfile`.

### 2. Set environment variables

In Railway → your service → **Variables**, add:

```
GROQ_API_KEY=your_groq_key
LIGHTNING_API_KEY=your_lightning_key   # optional fallback
```

### 3. Add a volume

In Railway → your service → **Volumes**, mount a persistent volume at `/app/storage` to persist your database and vector store between deploys.

### 4. Deploy

Railway builds and deploys automatically on every push to `main`.

---

## Deploy on Lightning AI (GPU)

Use this path if you want to run a local Qwen model alongside the app.

### 1. Create a Studio

Go to [lightning.ai](https://lightning.ai), create a new **Studio** with a GPU instance (L4 or A10G, 24 GB VRAM).

### 2. Clone and configure

```bash
git clone https://github.com/Jo-llama/ReqPal.git
cd ReqPal

cat > .env << 'ENV'
GROQ_API_KEY=your_groq_key
LITSERVE_ENABLED=1
QWEN_LOCAL_ENABLED=1
QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
ENV
```

### 3. Start

```bash
bash startup.sh
```

The startup script installs dependencies, starts the Qwen LitServe server on port 8000, and the FastAPI app on port 8001.

### 4. Access

In Lightning AI Studio, use **Port Forwarding** to expose port `8001`.

### VRAM Guide (local Qwen)

| Model | Float16 | 4-bit (`QWEN_LOAD_4BIT=1`) |
|---|---|---|
| Qwen2.5-7B-Instruct | ~15 GB | ~5 GB |
| Qwen2.5-14B-Instruct | ~30 GB | ~10 GB |

> When `GROQ_API_KEY` is set, Groq handles all LLM calls and the local model is only used as a fallback — you can skip the GPU entirely for production.

---

## Project Structure

```
ReqPal/
├── main.py                          # FastAPI app — all API endpoints
├── models.py                        # Pydantic models & enums
├── storage.py                       # SQLite CRUD — all entities
├── db_schema.py                     # SQLite schema (13 tables)
├── requirements.txt                 # Full dependencies (includes GPU/local LLM)
├── requirements-deploy.txt          # CPU-only dependencies (for Docker/Railway)
├── Dockerfile                       # CPU-only container for Railway
├── startup.sh                       # Lightning AI startup script
│
├── backend/
│   └── services/
│       ├── rag_service.py           # Chunking, embedding, ChromaDB retrieval
│       ├── reranker_service.py      # Cross-encoder reranking (lazy-loaded, CPU)
│       ├── llm_router.py            # Groq → Lightning → LitServe → Qwen → OpenAI → Ollama
│       └── rag_llm_prompts.py       # All LLM prompt templates
│
└── static/
    ├── index.html                   # PM Dashboard (tabbed, responsive)
    └── dashboard.html               # Stakeholder Dashboard (responsive)
```

---

## Architecture

| Layer | Technology |
|---|---|
| API | FastAPI |
| Persistence | SQLite |
| Vector Store | ChromaDB |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU, 22 MB) |
| LLM | Groq / Lightning AI / Qwen (local) / OpenAI / Ollama |
| Frontend | Vanilla JS, responsive, no build step |

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
