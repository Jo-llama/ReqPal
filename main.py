# main.py — ReqPal RAG MVP (SQLite backend + full CRUD)
# - Projects CRUD (+ delete cascade)
# - Documents upload/list/get/delete (Chroma cleanup)
# - /rag/search (chunks)
# - /rag/answer (rewrite -> retrieve -> filter -> optional LLM answer; ALWAYS returns context_chunks)
# - /providers-status (router debug)
# - Stakeholder requirements: full CRUD (PM + dashboard)
# - User stories: full CRUD (PM + dashboard)

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from pydantic import BaseModel
from typing import List as TList

import json
import csv as csv_module
import uuid
from io import StringIO

from models import (
    Project,
    Document,
    DocumentClassification,
    Requirement,
    UserStory,
    RAGQuery,
    Stakeholder,
    StakeholderRequirement,
    BulkImportResponse,
    DashboardChatRequest,
    DashboardChatResponse,
    DashboardRequirementSubmission,
)
from storage import storage
from backend.services.rag_service import rag_service
from backend.services.llm_router import LLMRouter
from backend.services.rag_llm_prompts import (
    QUERY_REWRITE_SYSTEM,
    ANSWER_SYSTEM,
    DASHBOARD_AGENT_SYSTEM,
    STORY_GENERATION_SYSTEM,
)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Pre-load Qwen model at startup so first request isn't slow
    try:
        r = LLMRouter()
        for p in r.providers:
            if hasattr(p, "warmup"):
                import threading
                threading.Thread(target=p.warmup, daemon=True).start()
                break
    except Exception as e:
        print(f"[startup] LLM warmup skipped: {e}")
    yield

app = FastAPI(title="ReqPal", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------------------
# LLM Router
# ----------------------------

router: Optional[LLMRouter] = None

def get_router() -> Optional[LLMRouter]:
    global router
    if router is not None:
        return router
    try:
        router = LLMRouter()
        return router
    except Exception:
        return None


# ----------------------------
# Helpers
# ----------------------------

def _safe_bool_env(key: str) -> bool:
    return bool((os.getenv(key) or "").strip())

def _project_or_404(project_id: int) -> Project:
    p = storage.get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return p

def _doc_or_404(document_id: int) -> Document:
    d = storage.get_document(document_id)
    if not d:
        raise HTTPException(404, "Document not found")
    return d

def _stakeholder_by_token(token: str) -> tuple:
    """Return (stakeholder, project) or raise 404."""
    s = storage.get_stakeholder_by_token(token)
    if not s:
        raise HTTPException(404, "Invalid dashboard link")
    p = storage.get_project(s.project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return s, p

def _filter_by_similarity(cands: List[Any], min_similarity: float) -> List[Any]:
    return [c for c in cands if float(getattr(c, "similarity_score", 0.0)) >= float(min_similarity)]

def _top_n(cands: List[Any], n: int) -> List[Any]:
    return list(cands[: max(0, n)])


# ----------------------------
# Routes: UI + Health
# ----------------------------

@app.get("/index.html")
async def index():
    path = os.path.join("static", "index.html")
    if not os.path.exists(path):
        raise HTTPException(404, "static/index.html not found")
    return FileResponse(path)

@app.get("/")
async def root():
    r = get_router()
    return {
        "status": "ReqPal RAG MVP is running",
        "version": app.version,
        "rag_enabled": True,
        "env": {
            "GROQ_API_KEY": _safe_bool_env("GROQ_API_KEY"),
            "OPENAI_API_KEY": _safe_bool_env("OPENAI_API_KEY"),
            "OLLAMA_BASE_URL": (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip(),
            "OLLAMA_MODEL": (os.getenv("OLLAMA_MODEL") or "").strip(),
        },
        "router": (r.providers_status() if r else {"configured": [], "models": {}}),
    }

@app.get("/providers-status")
async def providers_status():
    r = get_router()
    if not r:
        return {
            "ok": False,
            "error": "No LLM providers configured or router init failed.",
            "env": {
                "GROQ_API_KEY": _safe_bool_env("GROQ_API_KEY"),
                "OPENAI_API_KEY": _safe_bool_env("OPENAI_API_KEY"),
                "OLLAMA_BASE_URL": (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip(),
                "OLLAMA_MODEL": (os.getenv("OLLAMA_MODEL") or "").strip(),
            },
        }
    return {"ok": True, "router": r.providers_status()}


# ----------------------------
# Projects CRUD
# ----------------------------

@app.post("/projects", response_model=Project)
async def create_project(project: Project):
    project.id = storage.get_next_id("project")
    storage.add_project(project)
    return project

@app.get("/projects", response_model=List[Project])
async def list_projects():
    return storage.list_projects()

@app.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: int):
    return _project_or_404(project_id)

@app.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    _project_or_404(project_id)

    # Delete documents (get list first for cleanup)
    docs = storage.delete_documents_by_project(project_id)
    for d in docs:
        try:
            rag_service.delete_document_chunks(d.id or 0)
        except Exception:
            pass
        try:
            if d.file_path and os.path.exists(d.file_path):
                os.remove(d.file_path)
        except Exception:
            pass

    # Clean up vector store
    rag_service.delete_project_requirements(project_id)
    rag_service.delete_project_stories(project_id)

    # Delete related entities
    storage.delete_requirements_by_project(project_id)
    storage.delete_user_stories_by_project(project_id)
    storage.delete_stakeholder_requirements_by_project(project_id)
    storage.delete_stakeholders_by_project(project_id)

    storage.delete_project(project_id)
    return {"ok": True, "deleted_project_id": project_id, "deleted_documents": len(docs)}


# ----------------------------
# Documents
# ----------------------------

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    project_id: int = Form(...),
    classification: str = Form(...),
    document_purpose: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    _project_or_404(project_id)

    try:
        cls = DocumentClassification(classification)
    except Exception:
        raise HTTPException(400, f"Invalid classification: {classification}")

    content = await file.read()
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    doc_id = storage.get_next_id("document")

    try:
        document = await rag_service.process_document(
            file_content=content,
            filename=file.filename,
            project_id=project_id,
            classification=cls,
            document_id=doc_id,
            metadata={
                "document_purpose": document_purpose,
                "tags": tag_list,
            },
        )

        # Insert into SQLite (document already has id from doc_id)
        with storage._lock:
            storage._insert_document(document)
            storage._conn.commit()

        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded and processed successfully",
            "document": document,
            "chunks_created": len(document.chunks or []),
            "indexed": document.indexed,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@app.get("/documents")
async def list_documents(project_id: Optional[int] = None):
    return storage.list_documents(project_id)

@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    return _doc_or_404(document_id)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    d = _doc_or_404(document_id)

    try:
        rag_service.delete_document_chunks(d.id or 0)
    except Exception:
        pass

    try:
        if d.file_path and os.path.exists(d.file_path):
            os.remove(d.file_path)
    except Exception:
        pass

    storage.delete_document(document_id)
    return {"ok": True, "deleted_document_id": document_id}


# ----------------------------
# RAG: search (chunks)
# ----------------------------

@app.post("/rag/search")
async def rag_search(query: RAGQuery):
    _project_or_404(query.project_id)
    cands = await rag_service.semantic_search(query)
    return {
        "query": query.query,
        "project_id": query.project_id,
        "count": len(cands),
        "results": cands,
    }


# ----------------------------
# RAG: answer
# ----------------------------

class RAGAnswerRequest(BaseModel):
    query: str
    project_id: int
    top_k: int = 20
    min_similarity: float = 0.25
    document_classifications: TList[DocumentClassification] = []


@app.post("/rag/answer")
async def rag_answer(req: RAGAnswerRequest):
    project = _project_or_404(req.project_id)

    project_ctx = {
        "name": project.name,
        "description": project.description,
        "domain": getattr(project, "domain", None),
        "industry": getattr(project, "industry", None),
        "geography": getattr(project, "geography", []),
        "regulatory_exposure": getattr(project, "regulatory_exposure", []),
        "constraints": getattr(project, "constraints", None),
        "success_criteria": getattr(project, "success_criteria", []),
    }

    llm_trace_parts: List[str] = []
    rewritten_query = req.query
    r = get_router()

    # 1) Rewrite (best-effort)
    if r:
        try:
            rewrite_json, provider, _trace = await r.chat_json(
                system=QUERY_REWRITE_SYSTEM,
                user_payload={"project_context": project_ctx, "question": req.query},
                temperature=0.0,
                max_tokens=220,
                timeout_s=45.0,
                retries=1,
            )
            rewritten_query = (rewrite_json.get("rewritten_query") or req.query).strip()
            llm_trace_parts.append(f"rewrite:ok({provider})")
        except Exception as e:
            msg = str(e)
            if len(msg) > 120: msg = msg[:120] + "…"
            llm_trace_parts.append(f"rewrite:fail({type(e).__name__}:{msg})")
    else:
        llm_trace_parts.append("rewrite:skip(no_router)")

    # 2) Retrieve + rerank (multi-collection)
    selected = await rag_service.retrieve_and_rerank(
        query=rewritten_query,
        project_id=req.project_id,
        retrieval_mode="all",
        document_classifications=req.document_classifications or [],
        top_k=req.top_k,
    )

    # Fallback to original query if rewritten returned nothing
    if not selected and rewritten_query != req.query:
        selected = await rag_service.retrieve_and_rerank(
            query=req.query,
            project_id=req.project_id,
            retrieval_mode="all",
            document_classifications=req.document_classifications or [],
            top_k=req.top_k,
        )

    if not selected:
        return {
            "answer": ["No relevant context found in indexed documents."],
            "acceptance_criteria": [],
            "edge_cases": [],
            "open_questions": [
                "Which exact document contains the rule you are asking about?",
                "Upload the relevant policy/standard/audit evidence section (or narrow the question).",
            ],
            "citations_used": [],
            "context_chunks": [],
            "rewritten_query": rewritten_query,
            "llm_trace": " ".join(llm_trace_parts),
        }

    selected = selected[:8]
    llm_trace_parts.append(f"retrieve_rerank:{len(selected)}")
    top_ids = [c.chunk_id for c in selected]

    # 3) Answer (best-effort)
    answer_payload = {
        "answer": ["(LLM answer unavailable — returning retrieved context only.)"],
        "acceptance_criteria": [],
        "edge_cases": [],
        "open_questions": [],
        "citations_used": top_ids,
    }

    if r:
        try:
            context_chunks = [
                {
                    "chunk_id": c.chunk_id,
                    "document_name": c.document_name,
                    "classification": c.classification or c.category or "unknown",
                    "source_collection": c.source_collection.value if c.source_collection else None,
                    "content": (c.content or "")[:2200],
                }
                for c in selected
            ]

            ans_json, provider, _trace = await r.chat_json(
                system=ANSWER_SYSTEM,
                user_payload={
                    "question": req.query,
                    "project_context": project_ctx,
                    "context_chunks": context_chunks,
                },
                temperature=0.1,
                max_tokens=520,
                timeout_s=120.0,
                retries=1,
                backoff_s=1.0,
            )

            if isinstance(ans_json, dict):
                answer_payload = {
                    "answer": ans_json.get("answer", answer_payload["answer"]),
                    "acceptance_criteria": ans_json.get("acceptance_criteria", []),
                    "edge_cases": ans_json.get("edge_cases", []),
                    "open_questions": ans_json.get("open_questions", []),
                    "citations_used": ans_json.get("citations_used", top_ids) or top_ids,
                }

            llm_trace_parts.append(f"answer:ok({provider})")
        except Exception as e:
            msg = str(e)
            if len(msg) > 120: msg = msg[:120] + "…"
            llm_trace_parts.append(f"answer:fail({type(e).__name__}:{msg})")
    else:
        llm_trace_parts.append("answer:skip(no_router)")

    return {
        "answer": answer_payload.get("answer", []),
        "acceptance_criteria": answer_payload.get("acceptance_criteria", []),
        "edge_cases": answer_payload.get("edge_cases", []),
        "open_questions": answer_payload.get("open_questions", []),
        "citations_used": answer_payload.get("citations_used", top_ids),

        "context_chunks": [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "requirement_id": c.requirement_id,
                "document_name": c.document_name,
                "classification": c.classification,
                "source_collection": c.source_collection.value if c.source_collection else None,
                "content": c.content,
                "similarity_score": float(c.similarity_score),
                "page_number": c.page_number,
                "section": c.section,
                "stakeholder_role": c.stakeholder_role,
            }
            for c in selected
        ],
        "rewritten_query": rewritten_query,
        "llm_trace": " ".join(llm_trace_parts),
    }


# ----------------------------
# Requirements CRUD
# ----------------------------

@app.get("/requirements", response_model=List[Requirement])
async def list_requirements(project_id: Optional[int] = None):
    return storage.list_requirements(project_id)

@app.post("/requirements", response_model=Requirement)
async def create_requirement(req: Requirement):
    req.id = storage.get_next_id("requirement")
    storage.add_requirement(req)

    if req.project_id is not None:
        try:
            await rag_service.add_stakeholder_requirement(
                project_id=req.project_id,
                requirement_id=req.id,
                title=req.name,
                description=req.description,
                category=req.category,
                priority=req.priority,
                source=req.source,
            )
        except Exception as e:
            print(f"Warning: failed to index requirement in vector store: {e}")

    return req


# ----------------------------
# User Stories CRUD
# ----------------------------

@app.get("/user-stories", response_model=List[UserStory])
async def list_user_stories(project_id: Optional[int] = None):
    return storage.list_user_stories(project_id)

@app.post("/user-stories", response_model=UserStory)
async def create_user_story(story: UserStory):
    story.id = storage.get_next_id("story")
    storage.add_user_story(story)

    if story.project_id is not None:
        try:
            await rag_service.add_user_story(
                project_id=story.project_id,
                story_id=story.id,
                title=story.title,
                description=story.description,
                acceptance_criteria=story.acceptance_criteria or "",
                requirement_id=story.requirement_id,
                category=story.category,
            )
        except Exception as e:
            print(f"Warning: failed to index story in vector store: {e}")

    return story

@app.get("/user-stories/export")
async def export_user_stories(
    project_id: int = Query(...),
    format: str = Query("csv", regex="^(csv|json)$"),
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    _project_or_404(project_id)
    stories = storage.list_user_stories(project_id)

    if status:
        stories = [s for s in stories if s.status == status]
    if priority:
        stories = [s for s in stories if s.priority == priority]
    if category:
        stories = [s for s in stories if s.category == category]

    if format == "json":
        data = [s.dict() for s in stories]
        content = json.dumps(data, indent=2, default=str)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=user_stories_project_{project_id}.json"},
        )

    # CSV — Jira-compatible columns
    buf = StringIO()
    writer = csv_module.writer(buf)
    writer.writerow(["Summary", "Description", "Acceptance Criteria", "Priority", "Labels", "Status", "Issue Type"])
    for s in stories:
        writer.writerow([
            s.title,
            s.description,
            s.acceptance_criteria or "",
            s.priority or "",
            s.category or "",
            s.status or "",
            "Story",
        ])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=user_stories_project_{project_id}.csv"},
    )


@app.put("/user-stories/{story_id}", response_model=UserStory)
async def update_user_story(story_id: int, story: UserStory):
    existing = storage.get_user_story(story_id)
    if not existing:
        raise HTTPException(404, "User story not found")

    existing.title = story.title
    existing.description = story.description
    existing.acceptance_criteria = story.acceptance_criteria
    existing.category = story.category
    existing.priority = story.priority
    existing.status = story.status or existing.status
    storage.update_user_story(existing)

    if existing.project_id is not None:
        try:
            await rag_service.add_user_story(
                project_id=existing.project_id,
                story_id=existing.id,
                title=existing.title,
                description=existing.description,
                acceptance_criteria=existing.acceptance_criteria or "",
                requirement_id=existing.requirement_id,
                category=existing.category,
            )
        except Exception as e:
            print(f"Warning: failed to re-index story in vector store: {e}")

    return existing

@app.delete("/user-stories/{story_id}")
async def delete_user_story(story_id: int):
    existing = storage.get_user_story(story_id)
    if not existing:
        raise HTTPException(404, "User story not found")

    storage.delete_user_story(story_id)

    if existing.project_id is not None:
        try:
            rag_service.delete_user_story(existing.project_id, story_id)
        except Exception as e:
            print(f"Warning: failed to delete story from vector store: {e}")

    return {"ok": True, "deleted_story_id": story_id}


# ----------------------------
# PM-scoped stakeholder requirement CRUD
# ----------------------------

@app.put("/stakeholder-requirements/{req_id}")
async def update_stakeholder_requirement_pm(req_id: int, req: DashboardRequirementSubmission):
    sr = storage.get_stakeholder_requirement(req_id)
    if not sr:
        raise HTTPException(404, "Stakeholder requirement not found")

    sr.title = req.title
    sr.description = req.description
    sr.category = req.category
    sr.priority = req.priority
    sr.regulatory_mandate = req.regulatory_mandate
    sr.regulatory_references = req.regulatory_references
    storage.update_stakeholder_requirement(sr)

    try:
        await rag_service.add_stakeholder_requirement(
            project_id=sr.project_id,
            requirement_id=sr.id,
            title=sr.title,
            description=sr.description,
            stakeholder_role=sr.stakeholder_role,
            category=sr.category,
            priority=sr.priority,
            source=sr.source,
        )
    except Exception as e:
        print(f"Warning: failed to re-index requirement: {e}")

    return sr

@app.delete("/stakeholder-requirements/{req_id}")
async def delete_stakeholder_requirement_pm(req_id: int):
    sr = storage.get_stakeholder_requirement(req_id)
    if not sr:
        raise HTTPException(404, "Stakeholder requirement not found")

    storage.delete_stakeholder_requirement(req_id)

    try:
        rag_service.delete_stakeholder_requirement(sr.project_id, req_id)
    except Exception as e:
        print(f"Warning: failed to delete requirement from vector store: {e}")

    return {"ok": True, "deleted_requirement_id": req_id}


# ----------------------------
# Bulk Import: stakeholder requirements
# ----------------------------

@app.post("/api/projects/{project_id}/requirements/bulk-import")
async def bulk_import_requirements(
    project_id: int,
    file: UploadFile = File(...),
):
    _project_or_404(project_id)

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"
    import_batch_id = str(uuid.uuid4())[:8]

    items: List[dict] = []
    errors: List[str] = []

    if filename.endswith(".json"):
        try:
            raw = json.loads(text)
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, dict) and "requirements" in raw:
                items = raw["requirements"]
            else:
                raise ValueError("Expected a list or {requirements: [...]}")
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON: {e}")

    elif filename.endswith(".csv"):
        reader = csv_module.DictReader(StringIO(text))
        for row in reader:
            items.append(dict(row))
    else:
        raise HTTPException(400, "Unsupported file type. Use .json or .csv")

    imported: List[StakeholderRequirement] = []
    for i, item in enumerate(items):
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        stakeholder_role = (item.get("stakeholder_role") or "").strip()

        if not title or not description or not stakeholder_role:
            errors.append(f"Row {i}: missing required field (title, description, stakeholder_role)")
            continue

        req_id = storage.get_next_id("stakeholder_requirement")
        sr = StakeholderRequirement(
            id=req_id,
            project_id=project_id,
            title=title,
            description=description,
            stakeholder_role=stakeholder_role,
            category=item.get("category", "functional"),
            priority=item.get("priority", "Must Have"),
            source="bulk_import",
            import_batch_id=import_batch_id,
            regulatory_mandate=bool(item.get("regulatory_mandate", False)),
            regulatory_references=item.get("regulatory_references", []),
        )

        storage.add_stakeholder_requirement(sr)

        try:
            await rag_service.add_stakeholder_requirement(
                project_id=project_id,
                requirement_id=req_id,
                title=title,
                description=description,
                stakeholder_role=stakeholder_role,
                category=sr.category,
                priority=sr.priority,
                source="bulk_import",
                import_batch_id=import_batch_id,
            )
        except Exception as e:
            errors.append(f"Row {i}: vector indexing failed: {e}")

        imported.append(sr)

    return BulkImportResponse(
        success=len(errors) == 0,
        import_batch_id=import_batch_id,
        imported_count=len(imported),
        errors=errors,
        preview=imported[:10],
    )


# ----------------------------
# Stakeholder requirements: list
# ----------------------------

@app.get("/stakeholder-requirements")
async def list_stakeholder_requirements(project_id: Optional[int] = None):
    return storage.list_stakeholder_requirements(project_id)


# ----------------------------
# Stakeholder CRUD (PM-facing)
# ----------------------------

@app.post("/api/projects/{project_id}/stakeholders")
async def create_stakeholder(project_id: int, stakeholder: Stakeholder):
    _project_or_404(project_id)
    stakeholder.project_id = project_id
    stakeholder.id = storage.get_next_id("stakeholder")
    stakeholder.validation_token = uuid.uuid4().hex[:24]
    storage.add_stakeholder(stakeholder)
    return stakeholder

@app.get("/api/projects/{project_id}/stakeholders")
async def list_stakeholders(project_id: int):
    _project_or_404(project_id)
    return storage.list_stakeholders(project_id)

@app.delete("/api/projects/{project_id}/stakeholders/{stakeholder_id}")
async def delete_stakeholder(project_id: int, stakeholder_id: int):
    _project_or_404(project_id)
    s = storage.get_stakeholder(stakeholder_id)
    if not s or s.project_id != project_id:
        raise HTTPException(404, "Stakeholder not found")
    storage.delete_stakeholder(stakeholder_id)
    return {"ok": True, "deleted_stakeholder_id": stakeholder_id}


# ----------------------------
# Dashboard endpoints (stakeholder-facing, token-based)
# ----------------------------

@app.get("/dashboard/{token}")
async def serve_dashboard(token: str):
    _stakeholder_by_token(token)
    path = os.path.join("static", "dashboard.html")
    if not os.path.exists(path):
        raise HTTPException(404, "Dashboard page not found")
    return FileResponse(path)

@app.get("/api/dashboard/{token}")
async def get_dashboard_data(token: str):
    s, p = _stakeholder_by_token(token)

    my_reqs = storage.list_stakeholder_requirements(project_id=p.id, stakeholder_id=s.id)

    # All stories for this project
    all_stories = storage.list_user_stories(project_id=p.id)

    # Group stories by status
    pending_stories = [st.dict() for st in all_stories if st.status == "pending_review"]
    accepted_stories = [st.dict() for st in all_stories if st.stakeholder_validated]
    rejected_stories = [st.dict() for st in all_stories if st.status == "rejected"]

    # Relevant document summaries
    relevant_docs = []
    if s.relevant_document_classifications:
        class_values = [c.value if hasattr(c, "value") else c for c in s.relevant_document_classifications]
        for d in storage.list_documents(p.id):
            if d.classification.value in class_values:
                preview = ""
                if d.chunks:
                    preview = (d.chunks[0].content or "")[:300]
                relevant_docs.append({
                    "id": d.id,
                    "filename": d.filename,
                    "classification": d.classification.value,
                    "preview": preview,
                })

    return {
        "stakeholder": {
            "id": s.id,
            "name": s.name,
            "role": s.role,
            "type": s.type.value if hasattr(s.type, "value") else s.type,
            "concerns": s.concerns,
            "responsibilities": s.responsibilities,
        },
        "project": {
            "name": p.name,
            "description": p.description,
            "domain": p.domain,
            "regulatory_exposure": p.regulatory_exposure,
        },
        "my_requirements": [sr.dict() for sr in my_reqs],
        "pending_stories": pending_stories,
        "accepted_stories": accepted_stories,
        "rejected_stories": rejected_stories,
        "my_user_stories": [st.dict() for st in all_stories],
        "relevant_doc_summaries": relevant_docs,
    }

@app.post("/api/dashboard/{token}/chat", response_model=DashboardChatResponse)
async def dashboard_chat(token: str, req: DashboardChatRequest):
    s, p = _stakeholder_by_token(token)
    r = get_router()
    if not r:
        raise HTTPException(503, "No LLM providers available")

    project_ctx = (
        f"Project: {p.name}\n"
        f"Description: {p.description or 'N/A'}\n"
        f"Domain: {p.domain or 'N/A'}\n"
        f"Regulatory exposure: {', '.join(p.regulatory_exposure) if p.regulatory_exposure else 'None'}"
    )

    stakeholder_ctx = (
        f"Name: {s.name}\n"
        f"Role: {s.role}\n"
        f"Type: {s.type.value if hasattr(s.type, 'value') else s.type}\n"
        f"Concerns: {', '.join(s.concerns) if s.concerns else 'None'}\n"
        f"Responsibilities: {', '.join(s.responsibilities) if s.responsibilities else 'None'}"
    )

    compliance_ctx = "No compliance documents available."
    if req.messages and any(m.role == "user" for m in req.messages):
        first_user_msg = next((m.content for m in req.messages if m.role == "user"), "")
        try:
            doc_classes = s.relevant_document_classifications or []
            chunks = await rag_service.retrieve_context(
                query=first_user_msg,
                project_id=p.id,
                retrieval_mode="compliance",
                document_classifications=doc_classes,
                top_k=5,
            )
            if chunks:
                compliance_ctx = "\n---\n".join(
                    f"[{c.document_name} | {c.classification}]\n{c.content[:800]}"
                    for c in chunks
                )
        except Exception as e:
            print(f"Warning: dashboard RAG retrieval failed: {e}")

    system_prompt = DASHBOARD_AGENT_SYSTEM.format(
        project_context=project_ctx,
        stakeholder_context=stakeholder_ctx,
        compliance_context=compliance_ctx,
    )

    conversation = []
    for m in req.messages:
        conversation.append({"role": m.role, "content": m.content})

    try:
        result_json, provider, _trace = await r.chat_json(
            system=system_prompt,
            user_payload={"conversation": conversation},
            temperature=0.3,
            max_tokens=600,
            timeout_s=120.0,
            retries=1,
        )

        return DashboardChatResponse(
            reply=result_json.get("reply", "I'm sorry, I couldn't process that. Could you try again?"),
            finished=result_json.get("finished", False),
            draft_requirement=result_json.get("draft_requirement"),
        )
    except Exception as e:
        print(f"Dashboard chat error: {e}")
        raise HTTPException(503, f"LLM call failed: {str(e)[:200]}")

@app.post("/api/dashboard/{token}/requirements")
async def submit_dashboard_requirement(token: str, req: DashboardRequirementSubmission):
    s, p = _stakeholder_by_token(token)

    req_id = storage.get_next_id("stakeholder_requirement")
    sr = StakeholderRequirement(
        id=req_id,
        project_id=p.id,
        title=req.title,
        description=req.description,
        stakeholder_role=s.role,
        stakeholder_id=s.id,
        category=req.category,
        priority=req.priority,
        source="dashboard_chat",
        regulatory_mandate=req.regulatory_mandate,
        regulatory_references=req.regulatory_references,
    )

    storage.add_stakeholder_requirement(sr)

    # Index in ChromaDB
    try:
        await rag_service.add_stakeholder_requirement(
            project_id=p.id,
            requirement_id=req_id,
            title=req.title,
            description=req.description,
            stakeholder_role=s.role,
            category=req.category,
            priority=req.priority,
            source="dashboard_chat",
        )
    except Exception as e:
        print(f"Warning: failed to index dashboard requirement: {e}")

    # Duplicate detection
    potential_duplicates = []
    try:
        similar = await rag_service.search_similar_requirements(
            query=f"{req.title}\n{req.description}",
            project_id=p.id,
            top_k=5,
        )
        for sim in similar:
            if sim.requirement_id == req_id:
                continue
            if sim.similarity_score >= 0.65:
                potential_duplicates.append({
                    "requirement_id": sim.requirement_id,
                    "title": sim.document_name,
                    "similarity": round(sim.similarity_score, 3),
                    "content_preview": (sim.content or "")[:200],
                })
    except Exception as e:
        print(f"Warning: duplicate detection failed: {e}")

    if potential_duplicates:
        sr.metadata = sr.metadata or {}
        sr.metadata["potential_duplicates"] = potential_duplicates
        storage.update_stakeholder_requirement(sr)

    return {
        "requirement": sr.dict(),
        "potential_duplicates": potential_duplicates,
    }


@app.put("/api/dashboard/{token}/requirements/{requirement_id}")
async def update_dashboard_requirement(
    token: str, requirement_id: int, req: DashboardRequirementSubmission
):
    s, p = _stakeholder_by_token(token)

    sr = storage.get_stakeholder_requirement(requirement_id)
    if not sr or sr.project_id != p.id or sr.stakeholder_id != s.id:
        raise HTTPException(404, "Requirement not found")

    sr.title = req.title
    sr.description = req.description
    sr.category = req.category
    sr.priority = req.priority
    sr.regulatory_mandate = req.regulatory_mandate
    sr.regulatory_references = req.regulatory_references
    storage.update_stakeholder_requirement(sr)

    try:
        await rag_service.add_stakeholder_requirement(
            project_id=p.id,
            requirement_id=requirement_id,
            title=req.title,
            description=req.description,
            stakeholder_role=s.role,
            category=req.category,
            priority=req.priority,
            source=sr.source,
        )
    except Exception as e:
        print(f"Warning: failed to re-index requirement: {e}")

    return sr

@app.delete("/api/dashboard/{token}/requirements/{requirement_id}")
async def delete_dashboard_requirement(token: str, requirement_id: int):
    s, p = _stakeholder_by_token(token)

    sr = storage.get_stakeholder_requirement(requirement_id)
    if not sr or sr.project_id != p.id or sr.stakeholder_id != s.id:
        raise HTTPException(404, "Requirement not found")

    storage.delete_stakeholder_requirement(requirement_id)

    try:
        rag_service.delete_stakeholder_requirement(p.id, requirement_id)
    except Exception as e:
        print(f"Warning: failed to delete requirement from vector store: {e}")

    return {"ok": True, "deleted_requirement_id": requirement_id}


# ----------------------------
# Dashboard: Story Generation Agent
# ----------------------------

class GenerateStoriesRequest(BaseModel):
    requirement_ids: List[int]

@app.post("/api/dashboard/{token}/generate-stories")
async def generate_stories(token: str, req: GenerateStoriesRequest):
    s, p = _stakeholder_by_token(token)
    r = get_router()
    if not r:
        raise HTTPException(503, "No LLM providers available")

    if not req.requirement_ids:
        raise HTTPException(400, "requirement_ids must not be empty")

    project_ctx = (
        f"Project: {p.name}\n"
        f"Description: {p.description or 'N/A'}\n"
        f"Domain: {p.domain or 'N/A'}\n"
        f"Regulatory exposure: {', '.join(p.regulatory_exposure) if p.regulatory_exposure else 'None'}"
    )
    stakeholder_ctx = (
        f"Name: {s.name}\nRole: {s.role}\n"
        f"Type: {s.type.value if hasattr(s.type, 'value') else s.type}\n"
        f"Concerns: {', '.join(s.concerns) if s.concerns else 'None'}"
    )

    generated_by_req = {}

    for req_id in req.requirement_ids:
        sr = storage.get_stakeholder_requirement(req_id)
        if not sr or sr.project_id != p.id:
            continue

        # Retrieve compliance context via RAG
        compliance_ctx = "No compliance documents available."
        try:
            chunks = await rag_service.retrieve_and_rerank(
                query=f"{sr.title}\n{sr.description}",
                project_id=p.id,
                retrieval_mode="all",
                document_classifications=[],
                top_k=5,
            )
            if chunks:
                compliance_ctx = "\n---\n".join(
                    f"[{c.document_name} | {c.classification}]\n{(c.content or '')[:800]}"
                    for c in chunks[:5]
                )
        except Exception as e:
            print(f"Warning: RAG retrieval for story gen failed: {e}")

        system_prompt = STORY_GENERATION_SYSTEM.format(
            project_context=project_ctx,
            stakeholder_context=stakeholder_ctx,
            compliance_context=compliance_ctx,
        )

        try:
            result_json, provider, _trace = await r.chat_json(
                system=system_prompt,
                user_payload={
                    "requirement": {
                        "id": sr.id,
                        "title": sr.title,
                        "description": sr.description,
                        "category": sr.category,
                        "priority": sr.priority,
                        "stakeholder_role": sr.stakeholder_role,
                    }
                },
                temperature=0.3,
                max_tokens=1200,
                timeout_s=120.0,
                retries=1,
            )

            stories_data = result_json.get("stories", [])
            created_stories = []

            for story_data in stories_data:
                story_id = storage.get_next_id("story")
                story = UserStory(
                    id=story_id,
                    project_id=p.id,
                    requirement_id=sr.id,
                    stakeholder_id=s.id,
                    title=story_data.get("title", "Untitled Story"),
                    description=story_data.get("description", ""),
                    acceptance_criteria=story_data.get("acceptance_criteria", ""),
                    category=story_data.get("category", sr.category),
                    priority=story_data.get("priority", sr.priority),
                    status="pending_review",
                    stakeholder_validated=False,
                    pm_validated=False,
                )
                storage.add_user_story(story)

                # Index in ChromaDB
                try:
                    await rag_service.add_user_story(
                        project_id=p.id,
                        story_id=story_id,
                        title=story.title,
                        description=story.description,
                        acceptance_criteria=story.acceptance_criteria or "",
                        requirement_id=sr.id,
                        category=story.category,
                    )
                except Exception as e:
                    print(f"Warning: failed to index generated story: {e}")

                created_stories.append(story.dict())

            generated_by_req[str(req_id)] = {
                "requirement_title": sr.title,
                "stories": created_stories,
                "rationale": result_json.get("rationale", ""),
            }
        except Exception as e:
            print(f"Story generation failed for requirement {req_id}: {e}")
            generated_by_req[str(req_id)] = {
                "requirement_title": sr.title,
                "stories": [],
                "rationale": f"Generation failed: {str(e)[:200]}",
            }

    total = sum(len(v["stories"]) for v in generated_by_req.values())
    return {
        "success": True,
        "total_generated": total,
        "by_requirement": generated_by_req,
    }


@app.post("/api/dashboard/{token}/user-stories/{story_id}/accept")
async def accept_story(token: str, story_id: int):
    s, p = _stakeholder_by_token(token)
    story = storage.get_user_story(story_id)
    if not story or story.project_id != p.id:
        raise HTTPException(404, "User story not found")

    story.stakeholder_validated = True
    story.status = "accepted"
    storage.update_user_story(story)
    return story


@app.post("/api/dashboard/{token}/user-stories/{story_id}/reject")
async def reject_story(token: str, story_id: int):
    s, p = _stakeholder_by_token(token)
    story = storage.get_user_story(story_id)
    if not story or story.project_id != p.id:
        raise HTTPException(404, "User story not found")

    story.status = "rejected"
    storage.update_user_story(story)
    return story


# ----------------------------
# PM: Story Validation
# ----------------------------

@app.post("/user-stories/{story_id}/validate")
async def validate_story(story_id: int):
    story = storage.get_user_story(story_id)
    if not story:
        raise HTTPException(404, "User story not found")

    story.pm_validated = True
    story.status = "pm_validated"
    storage.update_user_story(story)
    return story


# ----------------------------
# Dashboard: User Stories CRUD (token-scoped)
# ----------------------------

class DashboardStorySubmission(BaseModel):
    title: str
    description: str
    acceptance_criteria: str = ""
    category: str = "functional"
    priority: str = "Must Have"

@app.get("/api/dashboard/{token}/user-stories")
async def list_dashboard_user_stories(token: str):
    s, p = _stakeholder_by_token(token)
    return storage.list_user_stories(project_id=p.id)

@app.post("/api/dashboard/{token}/user-stories")
async def create_dashboard_user_story(token: str, req: DashboardStorySubmission):
    s, p = _stakeholder_by_token(token)

    story_id = storage.get_next_id("story")
    story = UserStory(
        id=story_id,
        project_id=p.id,
        title=req.title,
        description=req.description,
        acceptance_criteria=req.acceptance_criteria,
        category=req.category,
        priority=req.priority,
        status="manual",
    )
    storage.add_user_story(story)

    try:
        await rag_service.add_user_story(
            project_id=p.id,
            story_id=story_id,
            title=req.title,
            description=req.description,
            acceptance_criteria=req.acceptance_criteria,
            category=req.category,
        )
    except Exception as e:
        print(f"Warning: failed to index user story: {e}")

    return story

@app.put("/api/dashboard/{token}/user-stories/{story_id}")
async def update_dashboard_user_story(token: str, story_id: int, req: DashboardStorySubmission):
    s, p = _stakeholder_by_token(token)

    story = storage.get_user_story(story_id)
    if not story or story.project_id != p.id:
        raise HTTPException(404, "User story not found")

    story.title = req.title
    story.description = req.description
    story.acceptance_criteria = req.acceptance_criteria
    story.category = req.category
    story.priority = req.priority
    storage.update_user_story(story)

    try:
        await rag_service.add_user_story(
            project_id=p.id,
            story_id=story_id,
            title=req.title,
            description=req.description,
            acceptance_criteria=req.acceptance_criteria,
            category=req.category,
        )
    except Exception as e:
        print(f"Warning: failed to re-index user story: {e}")

    return story

@app.delete("/api/dashboard/{token}/user-stories/{story_id}")
async def delete_dashboard_user_story(token: str, story_id: int):
    s, p = _stakeholder_by_token(token)

    story = storage.get_user_story(story_id)
    if not story or story.project_id != p.id:
        raise HTTPException(404, "User story not found")

    storage.delete_user_story(story_id)

    try:
        rag_service.delete_user_story(p.id, story_id)
    except Exception as e:
        print(f"Warning: failed to delete story from vector store: {e}")

    return {"ok": True, "deleted_story_id": story_id}


@app.get("/api/dashboard/{token}/user-stories/export")
async def export_dashboard_user_stories(
    token: str,
    format: str = Query("csv", regex="^(csv|json)$"),
):
    s, p = _stakeholder_by_token(token)
    stories = storage.list_user_stories(project_id=p.id)
    # Export only PM-validated (final) stories
    stories = [st for st in stories if st.pm_validated]

    if format == "json":
        data = [st.dict() for st in stories]
        content = json.dumps(data, indent=2, default=str)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=user_stories_{p.name.replace(' ', '_')}.json"},
        )

    buf = StringIO()
    writer = csv_module.writer(buf)
    writer.writerow(["Summary", "Description", "Acceptance Criteria", "Priority", "Labels", "Status", "Issue Type"])
    for st in stories:
        writer.writerow([
            st.title,
            st.description,
            st.acceptance_criteria or "",
            st.priority or "",
            st.category or "",
            st.status or "",
            "Story",
        ])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=user_stories_{p.name.replace(' ', '_')}.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
