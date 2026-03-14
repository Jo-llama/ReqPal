# storage.py — SQLite-backed storage for ReqPal RAG MVP
# Replaces the old JSON-file persistence with a thread-safe SQLite backend.

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from models import (
    Project, Requirement, UserStory, ProcessMap,
    KnowledgeGraph, ValidationToken,
    Document, DocumentChunk, Stakeholder, Risk, Assumption,
    GapAnalysis, TraceabilityLink,
    StakeholderRequirement, DocumentClassification,
)
from db_schema import CREATE_TABLES, SCHEMA_VERSION


def _json_dumps(obj) -> str:
    """Serialize Python object to JSON string for TEXT columns."""
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(val: Optional[str], default=None):
    """Deserialize JSON TEXT column back to Python object."""
    if val is None:
        return default if default is not None else None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else val


class Storage:
    """SQLite-backed storage with thread-safe access."""

    def __init__(self, db_path: str = "storage/reqpal.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Create tables
        self._conn.executescript(CREATE_TABLES)

        # Check schema version
        cur = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if not row:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self._conn.commit()

        # Add stakeholder_id column to user_stories if missing (schema migration)
        try:
            self._conn.execute("SELECT stakeholder_id FROM user_stories LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE user_stories ADD COLUMN stakeholder_id INTEGER")
            self._conn.commit()

        # Seed counter rows if empty
        counter_types = [
            "project", "requirement", "story", "process", "graph",
            "document", "stakeholder", "risk", "assumption",
            "gap_analysis", "traceability", "stakeholder_requirement",
        ]
        for ct in counter_types:
            self._conn.execute(
                "INSERT OR IGNORE INTO counters(entity_type, current_val) VALUES(?, 0)",
                (ct,),
            )
        self._conn.commit()

        # Migrate from JSON if applicable
        self._maybe_migrate_from_json()

        self._print_stats()

    # ------------------------------------------------------------------
    # JSON migration
    # ------------------------------------------------------------------

    def _maybe_migrate_from_json(self):
        json_path = "reqpal_data.json"
        if not os.path.exists(json_path):
            return

        # Only migrate if DB is empty
        cur = self._conn.execute("SELECT COUNT(*) FROM projects")
        if cur.fetchone()[0] > 0:
            return

        print(f"[MIGRATE] Found {json_path}, migrating to SQLite...")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read JSON for migration: {e}")
            return

        try:
            # Migrate projects
            for p in data.get("projects", []):
                proj = Project(**p)
                self._insert_project(proj)

            # Migrate documents
            for d in data.get("documents", []):
                doc = Document(**d)
                self._insert_document(doc)

            # Migrate requirements
            for r in data.get("requirements", []):
                req = Requirement(**r)
                self._insert_requirement(req)

            # Migrate stakeholder_requirements
            for sr in data.get("stakeholder_requirements", []):
                shreq = StakeholderRequirement(**sr)
                self._insert_stakeholder_requirement(shreq)

            # Migrate user_stories
            for s in data.get("user_stories", []):
                story = UserStory(**s)
                self._insert_user_story(story)

            # Migrate stakeholders
            for s in data.get("stakeholders", []):
                sh = Stakeholder(**s)
                self._insert_stakeholder(sh)

            # Migrate risks
            for r in data.get("risks", []):
                risk = Risk(**r)
                self._insert_risk(risk)

            # Migrate assumptions
            for a in data.get("assumptions", []):
                asmp = Assumption(**a)
                self._insert_assumption(asmp)

            # Migrate gap_analyses
            for ga in data.get("gap_analyses", []):
                gap = GapAnalysis(**ga)
                self._insert_gap_analysis(gap)

            # Migrate traceability_links
            for tl in data.get("traceability_links", []):
                link = TraceabilityLink(**tl)
                self._insert_traceability_link(link)

            # Migrate process_maps
            for pm in data.get("process_maps", []):
                proc = ProcessMap(**pm)
                self._insert_process_map(proc)

            # Migrate knowledge_graphs
            for kg in data.get("knowledge_graphs", []):
                graph = KnowledgeGraph(**kg)
                self._insert_knowledge_graph(graph)

            # Migrate validation_tokens
            for vt in data.get("validation_tokens", []):
                tok = ValidationToken(**vt)
                self._insert_validation_token(tok)

            # Rebuild counters from migrated data
            self._rebuild_counters()
            self._conn.commit()

            # Rename JSON file to .bak
            bak_path = json_path + ".bak"
            os.rename(json_path, bak_path)
            print(f"[MIGRATE] Migration complete. JSON renamed to {bak_path}")

        except Exception as e:
            print(f"[WARN] Migration failed: {e}")
            import traceback
            traceback.print_exc()

    def _rebuild_counters(self):
        """Rebuild counter values from max IDs in each table."""
        mapping = {
            "project": "SELECT COALESCE(MAX(id),0) FROM projects",
            "document": "SELECT COALESCE(MAX(id),0) FROM documents",
            "requirement": "SELECT COALESCE(MAX(id),0) FROM requirements",
            "stakeholder_requirement": "SELECT COALESCE(MAX(id),0) FROM stakeholder_requirements",
            "story": "SELECT COALESCE(MAX(id),0) FROM user_stories",
            "stakeholder": "SELECT COALESCE(MAX(id),0) FROM stakeholders",
            "risk": "SELECT COALESCE(MAX(id),0) FROM risks",
            "assumption": "SELECT COALESCE(MAX(id),0) FROM assumptions",
            "gap_analysis": "SELECT COALESCE(MAX(id),0) FROM gap_analyses",
            "traceability": "SELECT COALESCE(MAX(id),0) FROM traceability_links",
            "process": "SELECT COALESCE(MAX(id),0) FROM process_maps",
            "graph": "SELECT COALESCE(MAX(id),0) FROM knowledge_graphs",
        }
        for entity_type, sql in mapping.items():
            cur = self._conn.execute(sql)
            max_id = cur.fetchone()[0]
            self._conn.execute(
                "UPDATE counters SET current_val = MAX(current_val, ?) WHERE entity_type = ?",
                (max_id, entity_type),
            )

    # ------------------------------------------------------------------
    # Counter
    # ------------------------------------------------------------------

    def get_next_id(self, entity_type: str) -> int:
        with self._lock:
            self._conn.execute(
                "UPDATE counters SET current_val = current_val + 1 WHERE entity_type = ?",
                (entity_type,),
            )
            cur = self._conn.execute(
                "SELECT current_val FROM counters WHERE entity_type = ?",
                (entity_type,),
            )
            row = cur.fetchone()
            self._conn.commit()
            return row[0] if row else 1

    # ------------------------------------------------------------------
    # No-op save_data (backward compat)
    # ------------------------------------------------------------------

    def save_data(self):
        """No-op — each method auto-commits."""
        pass

    # ==================================================================
    # PROJECTS
    # ==================================================================

    def _insert_project(self, p: Project):
        self._conn.execute(
            """INSERT OR REPLACE INTO projects
               (id, name, description, domain, industry, geography,
                regulatory_exposure, intent, success_criteria, constraints,
                created_at, created_by, share_url, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                p.id, p.name, p.description, p.domain, p.industry,
                _json_dumps(p.geography), _json_dumps(p.regulatory_exposure),
                p.intent.value if hasattr(p.intent, "value") else p.intent,
                _json_dumps(p.success_criteria), _json_dumps(p.constraints),
                p.created_at, p.created_by, p.share_url,
                _json_dumps(p.metadata or {}),
            ),
        )

    def add_project(self, p: Project) -> Project:
        with self._lock:
            self._insert_project(p)
            self._conn.commit()
        return p

    def get_project(self, project_id: int) -> Optional[Project]:
        cur = self._conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cur.fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self) -> List[Project]:
        cur = self._conn.execute("SELECT * FROM projects ORDER BY id")
        return [self._row_to_project(r) for r in cur.fetchall()]

    def update_project(self, p: Project) -> None:
        with self._lock:
            self._insert_project(p)
            self._conn.commit()

    def delete_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self._conn.commit()

    def _row_to_project(self, row) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            domain=row["domain"],
            industry=row["industry"],
            geography=_json_loads(row["geography"], []),
            regulatory_exposure=_json_loads(row["regulatory_exposure"], []),
            intent=row["intent"] or "discovery",
            success_criteria=_json_loads(row["success_criteria"], []),
            constraints=_json_loads(row["constraints"], {"business": [], "legal": [], "technical": []}),
            created_at=row["created_at"] or datetime.now().isoformat(),
            created_by=row["created_by"],
            share_url=row["share_url"],
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # DOCUMENTS
    # ==================================================================

    def _insert_document(self, d: Document):
        chunks_json = _json_dumps([c.dict() for c in d.chunks]) if d.chunks else "[]"
        self._conn.execute(
            """INSERT OR REPLACE INTO documents
               (id, project_id, filename, file_type, file_path, file_size,
                classification, document_purpose, relevant_stakeholders, tags,
                uploaded_by, upload_date, processing_status, extraction_status,
                chunks, total_chunks, indexed, extracted_requirements, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d.id, d.project_id, d.filename, d.file_type, d.file_path, d.file_size,
                d.classification.value if hasattr(d.classification, "value") else d.classification,
                d.document_purpose,
                _json_dumps(d.relevant_stakeholders),
                _json_dumps(d.tags),
                d.uploaded_by, d.upload_date,
                d.processing_status, d.extraction_status,
                chunks_json,
                d.total_chunks, 1 if d.indexed else 0,
                _json_dumps(d.extracted_requirements),
                _json_dumps(d.metadata or {}),
            ),
        )

    def add_document(self, d: Document) -> Document:
        d.id = self.get_next_id("document")
        with self._lock:
            self._insert_document(d)
            self._conn.commit()
        return d

    def get_document(self, document_id: int) -> Optional[Document]:
        cur = self._conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self, project_id: Optional[int] = None) -> List[Document]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM documents ORDER BY id")
        return [self._row_to_document(r) for r in cur.fetchall()]

    def update_document(self, d: Document) -> None:
        with self._lock:
            self._insert_document(d)
            self._conn.commit()

    def delete_document(self, document_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            self._conn.commit()

    def delete_documents_by_project(self, project_id: int) -> List[Document]:
        """Delete all docs for a project. Returns the docs before deletion (for cleanup)."""
        docs = self.list_documents(project_id)
        with self._lock:
            self._conn.execute("DELETE FROM documents WHERE project_id = ?", (project_id,))
            self._conn.commit()
        return docs

    def _row_to_document(self, row) -> Document:
        chunks_raw = _json_loads(row["chunks"], [])
        chunks = [DocumentChunk(**c) for c in chunks_raw] if chunks_raw else []
        return Document(
            id=row["id"],
            project_id=row["project_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            file_path=row["file_path"],
            file_size=row["file_size"],
            classification=row["classification"],
            document_purpose=row["document_purpose"],
            relevant_stakeholders=_json_loads(row["relevant_stakeholders"], []),
            tags=_json_loads(row["tags"], []),
            uploaded_by=row["uploaded_by"],
            upload_date=row["upload_date"] or datetime.now().isoformat(),
            processing_status=row["processing_status"] or "pending",
            extraction_status=row["extraction_status"],
            chunks=chunks,
            total_chunks=row["total_chunks"] or 0,
            indexed=bool(row["indexed"]),
            extracted_requirements=_json_loads(row["extracted_requirements"], []),
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # REQUIREMENTS
    # ==================================================================

    def _insert_requirement(self, r: Requirement):
        self._conn.execute(
            """INSERT OR REPLACE INTO requirements
               (id, project_id, name, description, category, priority,
                source_documents, source_chunks, regulatory_mandate,
                regulatory_references, source, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r.id, r.project_id, r.name, r.description, r.category, r.priority,
                _json_dumps(r.source_documents), _json_dumps(r.source_chunks),
                1 if r.regulatory_mandate else 0,
                _json_dumps(r.regulatory_references),
                r.source, r.created_at, _json_dumps(r.metadata or {}),
            ),
        )

    def add_requirement(self, r: Requirement) -> Requirement:
        with self._lock:
            self._insert_requirement(r)
            self._conn.commit()
        return r

    def get_requirement(self, req_id: int) -> Optional[Requirement]:
        cur = self._conn.execute("SELECT * FROM requirements WHERE id = ?", (req_id,))
        row = cur.fetchone()
        return self._row_to_requirement(row) if row else None

    def list_requirements(self, project_id: Optional[int] = None) -> List[Requirement]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM requirements WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM requirements ORDER BY id")
        return [self._row_to_requirement(r) for r in cur.fetchall()]

    def update_requirement(self, r: Requirement) -> None:
        with self._lock:
            self._insert_requirement(r)
            self._conn.commit()

    def delete_requirement(self, req_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM requirements WHERE id = ?", (req_id,))
            self._conn.commit()

    def delete_requirements_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM requirements WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_requirement(self, row) -> Requirement:
        return Requirement(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"],
            category=row["category"] or "functional",
            priority=row["priority"] or "Must Have",
            source_documents=_json_loads(row["source_documents"], []),
            source_chunks=_json_loads(row["source_chunks"], []),
            regulatory_mandate=bool(row["regulatory_mandate"]),
            regulatory_references=_json_loads(row["regulatory_references"], []),
            source=row["source"] or "manual",
            created_at=row["created_at"] or datetime.now().isoformat(),
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # STAKEHOLDER REQUIREMENTS
    # ==================================================================

    def _insert_stakeholder_requirement(self, sr: StakeholderRequirement):
        self._conn.execute(
            """INSERT OR REPLACE INTO stakeholder_requirements
               (id, project_id, title, description, stakeholder_role, stakeholder_id,
                category, priority, source, import_batch_id,
                regulatory_mandate, regulatory_references, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sr.id, sr.project_id, sr.title, sr.description,
                sr.stakeholder_role, sr.stakeholder_id,
                sr.category, sr.priority, sr.source, sr.import_batch_id,
                1 if sr.regulatory_mandate else 0,
                _json_dumps(sr.regulatory_references),
                sr.created_at, _json_dumps(sr.metadata or {}),
            ),
        )

    def add_stakeholder_requirement(self, sr: StakeholderRequirement) -> StakeholderRequirement:
        with self._lock:
            self._insert_stakeholder_requirement(sr)
            self._conn.commit()
        return sr

    def get_stakeholder_requirement(self, req_id: int) -> Optional[StakeholderRequirement]:
        cur = self._conn.execute("SELECT * FROM stakeholder_requirements WHERE id = ?", (req_id,))
        row = cur.fetchone()
        return self._row_to_stakeholder_requirement(row) if row else None

    def list_stakeholder_requirements(self, project_id: Optional[int] = None, stakeholder_id: Optional[int] = None) -> List[StakeholderRequirement]:
        if project_id is not None and stakeholder_id is not None:
            cur = self._conn.execute(
                "SELECT * FROM stakeholder_requirements WHERE project_id = ? AND stakeholder_id = ? ORDER BY id",
                (project_id, stakeholder_id),
            )
        elif project_id is not None:
            cur = self._conn.execute(
                "SELECT * FROM stakeholder_requirements WHERE project_id = ? ORDER BY id",
                (project_id,),
            )
        else:
            cur = self._conn.execute("SELECT * FROM stakeholder_requirements ORDER BY id")
        return [self._row_to_stakeholder_requirement(r) for r in cur.fetchall()]

    def update_stakeholder_requirement(self, sr: StakeholderRequirement) -> None:
        with self._lock:
            self._insert_stakeholder_requirement(sr)
            self._conn.commit()

    def delete_stakeholder_requirement(self, req_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM stakeholder_requirements WHERE id = ?", (req_id,))
            self._conn.commit()

    def delete_stakeholder_requirements_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM stakeholder_requirements WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_stakeholder_requirement(self, row) -> StakeholderRequirement:
        return StakeholderRequirement(
            id=row["id"],
            project_id=row["project_id"],
            title=row["title"],
            description=row["description"],
            stakeholder_role=row["stakeholder_role"],
            stakeholder_id=row["stakeholder_id"],
            category=row["category"] or "functional",
            priority=row["priority"] or "Must Have",
            source=row["source"] or "manual",
            import_batch_id=row["import_batch_id"],
            regulatory_mandate=bool(row["regulatory_mandate"]),
            regulatory_references=_json_loads(row["regulatory_references"], []),
            created_at=row["created_at"] or datetime.now().isoformat(),
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # USER STORIES
    # ==================================================================

    def _insert_user_story(self, s: UserStory):
        self._conn.execute(
            """INSERT OR REPLACE INTO user_stories
               (id, project_id, requirement_id, stakeholder_id, title, description,
                acceptance_criteria, category, priority, source_documents,
                status, pm_validated, stakeholder_validated, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                s.id, s.project_id, s.requirement_id, s.stakeholder_id,
                s.title, s.description,
                s.acceptance_criteria or "", s.category, s.priority,
                _json_dumps(s.source_documents),
                s.status, 1 if s.pm_validated else 0,
                1 if s.stakeholder_validated else 0,
                s.created_at, _json_dumps(s.metadata or {}),
            ),
        )

    def add_user_story(self, s: UserStory) -> UserStory:
        with self._lock:
            self._insert_user_story(s)
            self._conn.commit()
        return s

    def get_user_story(self, story_id: int) -> Optional[UserStory]:
        cur = self._conn.execute("SELECT * FROM user_stories WHERE id = ?", (story_id,))
        row = cur.fetchone()
        return self._row_to_user_story(row) if row else None

    def list_user_stories(self, project_id: Optional[int] = None, stakeholder_id: Optional[int] = None) -> List[UserStory]:
        if project_id is not None and stakeholder_id is not None:
            cur = self._conn.execute(
                "SELECT * FROM user_stories WHERE project_id = ? AND stakeholder_id = ? ORDER BY id",
                (project_id, stakeholder_id),
            )
        elif project_id is not None:
            cur = self._conn.execute("SELECT * FROM user_stories WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM user_stories ORDER BY id")
        return [self._row_to_user_story(r) for r in cur.fetchall()]

    def update_user_story(self, s: UserStory) -> None:
        with self._lock:
            self._insert_user_story(s)
            self._conn.commit()

    def delete_user_story(self, story_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM user_stories WHERE id = ?", (story_id,))
            self._conn.commit()

    def delete_user_stories_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM user_stories WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_user_story(self, row) -> UserStory:
        # Handle both old schema (no stakeholder_id column) and new schema
        stakeholder_id = None
        try:
            stakeholder_id = row["stakeholder_id"]
        except (IndexError, KeyError):
            pass
        return UserStory(
            id=row["id"],
            project_id=row["project_id"],
            requirement_id=row["requirement_id"],
            stakeholder_id=stakeholder_id,
            title=row["title"],
            description=row["description"],
            acceptance_criteria=row["acceptance_criteria"] or "",
            category=row["category"] or "functional",
            priority=row["priority"] or "Must Have",
            source_documents=_json_loads(row["source_documents"], []),
            status=row["status"] or "ai_generated",
            pm_validated=bool(row["pm_validated"]),
            stakeholder_validated=bool(row["stakeholder_validated"]),
            created_at=row["created_at"] or datetime.now().isoformat(),
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # STAKEHOLDERS
    # ==================================================================

    def _insert_stakeholder(self, s: Stakeholder):
        rel_doc_class = [
            c.value if hasattr(c, "value") else c
            for c in (s.relevant_document_classifications or [])
        ]
        self._conn.execute(
            """INSERT OR REPLACE INTO stakeholders
               (id, project_id, name, email, role, type,
                responsibilities, concerns, relevant_document_classifications,
                relevant_requirement_categories, risk_areas, validation_token, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                s.id, s.project_id, s.name, s.email, s.role,
                s.type.value if hasattr(s.type, "value") else s.type,
                _json_dumps(s.responsibilities),
                _json_dumps(s.concerns),
                _json_dumps(rel_doc_class),
                _json_dumps(s.relevant_requirement_categories),
                _json_dumps(s.risk_areas),
                s.validation_token,
                _json_dumps(s.metadata or {}),
            ),
        )

    def add_stakeholder(self, s: Stakeholder) -> Stakeholder:
        with self._lock:
            self._insert_stakeholder(s)
            self._conn.commit()
        return s

    def get_stakeholder(self, stakeholder_id: int) -> Optional[Stakeholder]:
        cur = self._conn.execute("SELECT * FROM stakeholders WHERE id = ?", (stakeholder_id,))
        row = cur.fetchone()
        return self._row_to_stakeholder(row) if row else None

    def get_stakeholder_by_token(self, token: str) -> Optional[Stakeholder]:
        cur = self._conn.execute("SELECT * FROM stakeholders WHERE validation_token = ?", (token,))
        row = cur.fetchone()
        return self._row_to_stakeholder(row) if row else None

    def list_stakeholders(self, project_id: Optional[int] = None) -> List[Stakeholder]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM stakeholders WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM stakeholders ORDER BY id")
        return [self._row_to_stakeholder(r) for r in cur.fetchall()]

    def update_stakeholder(self, s: Stakeholder) -> None:
        with self._lock:
            self._insert_stakeholder(s)
            self._conn.commit()

    def delete_stakeholder(self, stakeholder_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM stakeholders WHERE id = ?", (stakeholder_id,))
            self._conn.commit()

    def delete_stakeholders_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM stakeholders WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_stakeholder(self, row) -> Stakeholder:
        raw_classes = _json_loads(row["relevant_document_classifications"], [])
        doc_classes = []
        for c in raw_classes:
            try:
                doc_classes.append(DocumentClassification(c))
            except (ValueError, KeyError):
                pass
        return Stakeholder(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            email=row["email"],
            role=row["role"],
            type=row["type"],
            responsibilities=_json_loads(row["responsibilities"], []),
            concerns=_json_loads(row["concerns"], []),
            relevant_document_classifications=doc_classes,
            relevant_requirement_categories=_json_loads(row["relevant_requirement_categories"], []),
            risk_areas=_json_loads(row["risk_areas"], []),
            validation_token=row["validation_token"],
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # RISKS
    # ==================================================================

    def _insert_risk(self, r: Risk):
        self._conn.execute(
            """INSERT OR REPLACE INTO risks
               (id, project_id, requirement_id, document_id, risk_type, severity,
                title, description, impact, likelihood, mitigation, owner,
                status, source_document_chunks, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r.id, r.project_id, r.requirement_id, r.document_id,
                r.risk_type, r.severity, r.title, r.description,
                r.impact, r.likelihood, r.mitigation, r.owner,
                r.status, _json_dumps(r.source_document_chunks), r.created_at,
            ),
        )

    def add_risk(self, r: Risk) -> Risk:
        with self._lock:
            self._insert_risk(r)
            self._conn.commit()
        return r

    def list_risks(self, project_id: Optional[int] = None) -> List[Risk]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM risks WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM risks ORDER BY id")
        return [self._row_to_risk(r) for r in cur.fetchall()]

    def delete_risks_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM risks WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_risk(self, row) -> Risk:
        return Risk(
            id=row["id"],
            project_id=row["project_id"],
            requirement_id=row["requirement_id"],
            document_id=row["document_id"],
            risk_type=row["risk_type"],
            severity=row["severity"],
            title=row["title"],
            description=row["description"],
            impact=row["impact"],
            likelihood=row["likelihood"],
            mitigation=row["mitigation"],
            owner=row["owner"],
            status=row["status"] or "identified",
            source_document_chunks=_json_loads(row["source_document_chunks"], []),
            created_at=row["created_at"] or datetime.now().isoformat(),
        )

    # ==================================================================
    # ASSUMPTIONS
    # ==================================================================

    def _insert_assumption(self, a: Assumption):
        self._conn.execute(
            """INSERT OR REPLACE INTO assumptions
               (id, project_id, requirement_id, assumption, validation_status,
                impact_if_wrong, validation_method, owner, created_at, validated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                a.id, a.project_id, a.requirement_id, a.assumption,
                a.validation_status, a.impact_if_wrong, a.validation_method,
                a.owner, a.created_at, a.validated_at,
            ),
        )

    def add_assumption(self, a: Assumption) -> Assumption:
        with self._lock:
            self._insert_assumption(a)
            self._conn.commit()
        return a

    def list_assumptions(self, project_id: Optional[int] = None) -> List[Assumption]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM assumptions WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM assumptions ORDER BY id")
        return [self._row_to_assumption(r) for r in cur.fetchall()]

    def delete_assumptions_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM assumptions WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_assumption(self, row) -> Assumption:
        return Assumption(
            id=row["id"],
            project_id=row["project_id"],
            requirement_id=row["requirement_id"],
            assumption=row["assumption"],
            validation_status=row["validation_status"] or "unvalidated",
            impact_if_wrong=row["impact_if_wrong"],
            validation_method=row["validation_method"],
            owner=row["owner"],
            created_at=row["created_at"] or datetime.now().isoformat(),
            validated_at=row["validated_at"],
        )

    # ==================================================================
    # GAP ANALYSES
    # ==================================================================

    def _insert_gap_analysis(self, ga: GapAnalysis):
        gaps_json = _json_dumps([g.dict() for g in ga.identified_gaps]) if ga.identified_gaps else "[]"
        self._conn.execute(
            """INSERT OR REPLACE INTO gap_analyses
               (id, project_id, analysis_date, identified_gaps,
                missing_documents, unclear_areas, recommendations, coverage_score)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                ga.id, ga.project_id, ga.analysis_date, gaps_json,
                _json_dumps(ga.missing_documents),
                _json_dumps(ga.unclear_areas),
                _json_dumps(ga.recommendations),
                ga.coverage_score,
            ),
        )

    def add_gap_analysis(self, ga: GapAnalysis) -> GapAnalysis:
        with self._lock:
            self._insert_gap_analysis(ga)
            self._conn.commit()
        return ga

    def list_gap_analyses(self, project_id: Optional[int] = None) -> List[GapAnalysis]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM gap_analyses WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM gap_analyses ORDER BY id")
        return [self._row_to_gap_analysis(r) for r in cur.fetchall()]

    def delete_gap_analyses_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM gap_analyses WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_gap_analysis(self, row) -> GapAnalysis:
        from models import Gap
        raw_gaps = _json_loads(row["identified_gaps"], [])
        gaps = [Gap(**g) for g in raw_gaps] if raw_gaps else []
        return GapAnalysis(
            id=row["id"],
            project_id=row["project_id"],
            analysis_date=row["analysis_date"] or datetime.now().isoformat(),
            identified_gaps=gaps,
            missing_documents=_json_loads(row["missing_documents"], []),
            unclear_areas=_json_loads(row["unclear_areas"], []),
            recommendations=_json_loads(row["recommendations"], []),
            coverage_score=row["coverage_score"] or 0.0,
        )

    # ==================================================================
    # TRACEABILITY LINKS
    # ==================================================================

    def _insert_traceability_link(self, tl: TraceabilityLink):
        self._conn.execute(
            """INSERT OR REPLACE INTO traceability_links
               (id, project_id, source_type, source_id, target_type, target_id,
                relationship, rationale, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                tl.id, tl.project_id, tl.source_type, tl.source_id,
                tl.target_type, tl.target_id, tl.relationship,
                tl.rationale, tl.created_at,
            ),
        )

    def add_traceability_link(self, tl: TraceabilityLink) -> TraceabilityLink:
        with self._lock:
            self._insert_traceability_link(tl)
            self._conn.commit()
        return tl

    def list_traceability_links(self, project_id: Optional[int] = None) -> List[TraceabilityLink]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM traceability_links WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM traceability_links ORDER BY id")
        return [self._row_to_traceability_link(r) for r in cur.fetchall()]

    def delete_traceability_links_by_project(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM traceability_links WHERE project_id = ?", (project_id,))
            self._conn.commit()

    def _row_to_traceability_link(self, row) -> TraceabilityLink:
        return TraceabilityLink(
            id=row["id"],
            project_id=row["project_id"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            target_type=row["target_type"],
            target_id=row["target_id"],
            relationship=row["relationship"],
            rationale=row["rationale"],
            created_at=row["created_at"] or datetime.now().isoformat(),
        )

    # ==================================================================
    # PROCESS MAPS
    # ==================================================================

    def _insert_process_map(self, pm: ProcessMap):
        self._conn.execute(
            """INSERT OR REPLACE INTO process_maps
               (id, project_id, name, type, steps, created_at, metadata)
               VALUES (?,?,?,?,?,?,?)""",
            (
                pm.id, pm.project_id, pm.name, pm.type,
                _json_dumps(pm.steps), pm.created_at,
                _json_dumps(pm.metadata or {}),
            ),
        )

    def add_process_map(self, pm: ProcessMap) -> ProcessMap:
        with self._lock:
            self._insert_process_map(pm)
            self._conn.commit()
        return pm

    def list_process_maps(self, project_id: Optional[int] = None) -> List[ProcessMap]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM process_maps WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM process_maps ORDER BY id")
        return [self._row_to_process_map(r) for r in cur.fetchall()]

    def _row_to_process_map(self, row) -> ProcessMap:
        return ProcessMap(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            type=row["type"] or "bpmn",
            steps=_json_loads(row["steps"], []),
            created_at=row["created_at"] or datetime.now().isoformat(),
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # KNOWLEDGE GRAPHS
    # ==================================================================

    def _insert_knowledge_graph(self, kg: KnowledgeGraph):
        nodes_json = _json_dumps([n.dict() for n in kg.nodes]) if kg.nodes else "[]"
        edges_json = _json_dumps([e.dict() for e in kg.edges]) if kg.edges else "[]"
        self._conn.execute(
            """INSERT OR REPLACE INTO knowledge_graphs
               (id, project_id, nodes, edges, created_at, updated_at, metadata)
               VALUES (?,?,?,?,?,?,?)""",
            (
                kg.id, kg.project_id, nodes_json, edges_json,
                kg.created_at, kg.updated_at, _json_dumps(kg.metadata or {}),
            ),
        )

    def add_knowledge_graph(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        with self._lock:
            self._insert_knowledge_graph(kg)
            self._conn.commit()
        return kg

    def list_knowledge_graphs(self, project_id: Optional[int] = None) -> List[KnowledgeGraph]:
        if project_id is not None:
            cur = self._conn.execute("SELECT * FROM knowledge_graphs WHERE project_id = ? ORDER BY id", (project_id,))
        else:
            cur = self._conn.execute("SELECT * FROM knowledge_graphs ORDER BY id")
        return [self._row_to_knowledge_graph(r) for r in cur.fetchall()]

    def _row_to_knowledge_graph(self, row) -> KnowledgeGraph:
        from models import KnowledgeGraphNode, KnowledgeGraphEdge
        raw_nodes = _json_loads(row["nodes"], [])
        raw_edges = _json_loads(row["edges"], [])
        return KnowledgeGraph(
            id=row["id"],
            project_id=row["project_id"],
            nodes=[KnowledgeGraphNode(**n) for n in raw_nodes] if raw_nodes else [],
            edges=[KnowledgeGraphEdge(**e) for e in raw_edges] if raw_edges else [],
            created_at=row["created_at"] or datetime.now().isoformat(),
            updated_at=row["updated_at"] or datetime.now().isoformat(),
            metadata=_json_loads(row["metadata"], {}),
        )

    # ==================================================================
    # VALIDATION TOKENS
    # ==================================================================

    def _insert_validation_token(self, vt: ValidationToken):
        self._conn.execute(
            """INSERT OR REPLACE INTO validation_tokens
               (token, project_id, stakeholder_email, created_at, expires_at, used)
               VALUES (?,?,?,?,?,?)""",
            (
                vt.token, vt.project_id, vt.stakeholder_email,
                vt.created_at, vt.expires_at, 1 if vt.used else 0,
            ),
        )

    def add_validation_token(self, vt: ValidationToken) -> ValidationToken:
        with self._lock:
            self._insert_validation_token(vt)
            self._conn.commit()
        return vt

    def list_validation_tokens(self) -> List[ValidationToken]:
        cur = self._conn.execute("SELECT * FROM validation_tokens")
        return [self._row_to_validation_token(r) for r in cur.fetchall()]

    def _row_to_validation_token(self, row) -> ValidationToken:
        return ValidationToken(
            token=row["token"],
            project_id=row["project_id"],
            stakeholder_email=row["stakeholder_email"],
            created_at=row["created_at"] or datetime.now().isoformat(),
            expires_at=row["expires_at"],
            used=bool(row["used"]),
        )

    # ==================================================================
    # STATS
    # ==================================================================

    def get_stats(self) -> Dict[str, Any]:
        tables = [
            ("projects", "projects"),
            ("requirements", "requirements"),
            ("user_stories", "user_stories"),
            ("process_maps", "process_maps"),
            ("knowledge_graphs", "knowledge_graphs"),
            ("validation_tokens", "validation_tokens"),
            ("documents", "documents"),
            ("stakeholders", "stakeholders"),
            ("risks", "risks"),
            ("assumptions", "assumptions"),
            ("gap_analyses", "gap_analyses"),
            ("traceability_links", "traceability_links"),
            ("stakeholder_requirements", "stakeholder_requirements"),
        ]
        stats = {}
        for key, table in tables:
            cur = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[key] = cur.fetchone()[0]
        return stats

    def _print_stats(self):
        stats = self.get_stats()
        total = sum(stats.values())
        print(f"[OK] SQLite storage loaded from {self.db_path}")
        if total > 0:
            for key, count in stats.items():
                if count > 0:
                    print(f"   - {key}: {count}")


# Singleton
storage = Storage()
__all__ = ["storage", "Storage"]
