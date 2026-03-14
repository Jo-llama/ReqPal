# db_schema.py — SQLite table definitions for ReqPal RAG MVP

SCHEMA_VERSION = 1

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS counters (
    entity_type TEXT PRIMARY KEY,
    current_val INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS projects (
    id                   INTEGER PRIMARY KEY,
    name                 TEXT NOT NULL,
    description          TEXT,
    domain               TEXT,
    industry             TEXT,
    geography            TEXT DEFAULT '[]',
    regulatory_exposure  TEXT DEFAULT '[]',
    intent               TEXT DEFAULT 'discovery',
    success_criteria     TEXT DEFAULT '[]',
    constraints          TEXT DEFAULT '{}',
    created_at           TEXT,
    created_by           TEXT,
    share_url            TEXT,
    metadata             TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS documents (
    id                     INTEGER PRIMARY KEY,
    project_id             INTEGER NOT NULL,
    filename               TEXT NOT NULL,
    file_type              TEXT NOT NULL,
    file_path              TEXT NOT NULL,
    file_size              INTEGER NOT NULL,
    classification         TEXT NOT NULL,
    document_purpose       TEXT,
    relevant_stakeholders  TEXT DEFAULT '[]',
    tags                   TEXT DEFAULT '[]',
    uploaded_by            TEXT,
    upload_date            TEXT,
    processing_status      TEXT DEFAULT 'pending',
    extraction_status      TEXT,
    chunks                 TEXT DEFAULT '[]',
    total_chunks           INTEGER DEFAULT 0,
    indexed                INTEGER DEFAULT 0,
    extracted_requirements TEXT DEFAULT '[]',
    metadata               TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);

CREATE TABLE IF NOT EXISTS requirements (
    id                    INTEGER PRIMARY KEY,
    project_id            INTEGER,
    name                  TEXT NOT NULL,
    description           TEXT NOT NULL,
    category              TEXT DEFAULT 'functional',
    priority              TEXT DEFAULT 'Must Have',
    source_documents      TEXT DEFAULT '[]',
    source_chunks         TEXT DEFAULT '[]',
    regulatory_mandate    INTEGER DEFAULT 0,
    regulatory_references TEXT DEFAULT '[]',
    source                TEXT DEFAULT 'manual',
    created_at            TEXT,
    metadata              TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_requirements_project ON requirements(project_id);

CREATE TABLE IF NOT EXISTS stakeholder_requirements (
    id                    INTEGER PRIMARY KEY,
    project_id            INTEGER NOT NULL,
    title                 TEXT NOT NULL,
    description           TEXT NOT NULL,
    stakeholder_role      TEXT NOT NULL,
    stakeholder_id        INTEGER,
    category              TEXT DEFAULT 'functional',
    priority              TEXT DEFAULT 'Must Have',
    source                TEXT DEFAULT 'manual',
    import_batch_id       TEXT,
    regulatory_mandate    INTEGER DEFAULT 0,
    regulatory_references TEXT DEFAULT '[]',
    created_at            TEXT,
    metadata              TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_shreqs_project ON stakeholder_requirements(project_id);
CREATE INDEX IF NOT EXISTS idx_shreqs_stakeholder ON stakeholder_requirements(stakeholder_id);

CREATE TABLE IF NOT EXISTS user_stories (
    id                   INTEGER PRIMARY KEY,
    project_id           INTEGER,
    requirement_id       INTEGER,
    stakeholder_id       INTEGER,
    title                TEXT NOT NULL,
    description          TEXT NOT NULL,
    acceptance_criteria  TEXT DEFAULT '',
    category             TEXT DEFAULT 'functional',
    priority             TEXT DEFAULT 'Must Have',
    source_documents     TEXT DEFAULT '[]',
    status               TEXT DEFAULT 'ai_generated',
    pm_validated         INTEGER DEFAULT 0,
    stakeholder_validated INTEGER DEFAULT 0,
    created_at           TEXT,
    metadata             TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_stories_project ON user_stories(project_id);

CREATE TABLE IF NOT EXISTS stakeholders (
    id                                INTEGER PRIMARY KEY,
    project_id                        INTEGER NOT NULL,
    name                              TEXT NOT NULL,
    email                             TEXT,
    role                              TEXT NOT NULL,
    type                              TEXT NOT NULL,
    responsibilities                  TEXT DEFAULT '[]',
    concerns                          TEXT DEFAULT '[]',
    relevant_document_classifications TEXT DEFAULT '[]',
    relevant_requirement_categories   TEXT DEFAULT '[]',
    risk_areas                        TEXT DEFAULT '[]',
    validation_token                  TEXT,
    metadata                          TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_stakeholders_project ON stakeholders(project_id);

CREATE TABLE IF NOT EXISTS risks (
    id                     INTEGER PRIMARY KEY,
    project_id             INTEGER NOT NULL,
    requirement_id         INTEGER,
    document_id            INTEGER,
    risk_type              TEXT NOT NULL,
    severity               TEXT NOT NULL,
    title                  TEXT NOT NULL,
    description            TEXT NOT NULL,
    impact                 TEXT NOT NULL,
    likelihood             TEXT NOT NULL,
    mitigation             TEXT,
    owner                  TEXT,
    status                 TEXT DEFAULT 'identified',
    source_document_chunks TEXT DEFAULT '[]',
    created_at             TEXT
);
CREATE INDEX IF NOT EXISTS idx_risks_project ON risks(project_id);

CREATE TABLE IF NOT EXISTS assumptions (
    id                INTEGER PRIMARY KEY,
    project_id        INTEGER NOT NULL,
    requirement_id    INTEGER,
    assumption        TEXT NOT NULL,
    validation_status TEXT DEFAULT 'unvalidated',
    impact_if_wrong   TEXT NOT NULL,
    validation_method TEXT,
    owner             TEXT,
    created_at        TEXT,
    validated_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_assumptions_project ON assumptions(project_id);

CREATE TABLE IF NOT EXISTS gap_analyses (
    id                 INTEGER PRIMARY KEY,
    project_id         INTEGER NOT NULL,
    analysis_date      TEXT,
    identified_gaps    TEXT DEFAULT '[]',
    missing_documents  TEXT DEFAULT '[]',
    unclear_areas      TEXT DEFAULT '[]',
    recommendations    TEXT DEFAULT '[]',
    coverage_score     REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_gap_project ON gap_analyses(project_id);

CREATE TABLE IF NOT EXISTS traceability_links (
    id           INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL,
    source_type  TEXT NOT NULL,
    source_id    INTEGER NOT NULL,
    target_type  TEXT NOT NULL,
    target_id    INTEGER NOT NULL,
    relationship TEXT NOT NULL,
    rationale    TEXT,
    created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_project ON traceability_links(project_id);

CREATE TABLE IF NOT EXISTS process_maps (
    id         INTEGER PRIMARY KEY,
    project_id INTEGER,
    name       TEXT NOT NULL,
    type       TEXT DEFAULT 'bpmn',
    steps      TEXT DEFAULT '[]',
    created_at TEXT,
    metadata   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS knowledge_graphs (
    id         INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL,
    nodes      TEXT DEFAULT '[]',
    edges      TEXT DEFAULT '[]',
    created_at TEXT,
    updated_at TEXT,
    metadata   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS validation_tokens (
    token            TEXT PRIMARY KEY,
    project_id       INTEGER NOT NULL,
    stakeholder_email TEXT,
    created_at       TEXT,
    expires_at       TEXT,
    used             INTEGER DEFAULT 0
);
"""
