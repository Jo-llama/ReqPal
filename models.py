# models.py - UPDATED v3.3 with RAG Support
# Enhanced Project Creation + Document Management

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ==================== ENUMS ====================

class ProjectIntent(str, Enum):
    DISCOVERY = "discovery"
    TRANSFORMATION = "transformation"
    COMPLIANCE = "compliance"
    API_DEVELOPMENT = "api_development"
    HYBRID = "hybrid"

class DocumentClassification(str, Enum):
    REGULATION = "regulation"
    PROCESS = "process"
    EVIDENCE = "evidence"
    DISCOVERY = "discovery"
    DECISION = "decision"
    SPECIFICATION = "specification"
    LEGAL = "legal"
    AUDIT = "audit"

class StakeholderType(str, Enum):
    PRODUCT = "product"
    BUSINESS = "business"
    CONTROL = "control"
    EXTERNAL = "external"

class CollectionName(str, Enum):
    COMPLIANCE = "compliance_documents"
    REQUIREMENTS = "stakeholder_requirements"
    USER_STORIES = "user_stories"

class RetrievalMode(str, Enum):
    ALL = "all"
    COMPLIANCE = "compliance"
    REQUIREMENTS = "requirements"

# ==================== ENHANCED PROJECT ====================

class Project(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    
    # NEW: Rich Metadata for RAG
    domain: Optional[str] = None  # "Banking", "Healthcare", "E-commerce"
    industry: Optional[str] = None
    geography: List[str] = []  # ["EU", "US", "APAC"]
    regulatory_exposure: List[str] = []  # ["GDPR", "HIPAA", "SOX", "PCI-DSS"]
    
    # Project Intent
    intent: ProjectIntent = ProjectIntent.DISCOVERY
    
    # Success Criteria & Constraints
    success_criteria: List[str] = []
    constraints: Dict[str, List[str]] = {
        "business": [],
        "legal": [],
        "technical": []
    }
    
    # Metadata
    created_at: str = datetime.now().isoformat()
    created_by: Optional[str] = None
    share_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

# ==================== DOCUMENT MANAGEMENT ====================

class DocumentChunk(BaseModel):
    """Chunked text for RAG retrieval"""
    chunk_id: str
    document_id: int
    content: str
    chunk_index: int
    
    # Metadata for retrieval
    page_number: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    
    # Vector embedding (stored as ID reference to vector DB)
    embedding_id: Optional[str] = None
    
    # Metadata for filtering
    metadata: Dict[str, Any] = {}

class Document(BaseModel):
    """Uploaded document with RAG support"""
    id: Optional[int] = None
    project_id: int
    filename: str
    file_type: str  # "pdf", "docx", "csv", "json", "bpmn", "txt"
    file_path: str  # Storage path
    file_size: int  # bytes
    
    # Classification for RAG
    classification: DocumentClassification
    
    # Context metadata (for retrieval filtering)
    document_purpose: Optional[str] = None  # Why was this uploaded?
    relevant_stakeholders: List[str] = []  # Who cares about this?
    tags: List[str] = []
    
    # Upload metadata
    uploaded_by: Optional[str] = None
    upload_date: str = datetime.now().isoformat()
    
    # Processing status
    processing_status: str = "pending"  # pending, processing, completed, failed
    extraction_status: Optional[str] = None
    
    # RAG-specific
    chunks: List[DocumentChunk] = []
    total_chunks: int = 0
    indexed: bool = False
    
    # Extracted requirements (if any)
    extracted_requirements: List[int] = []  # Requirement IDs
    
    metadata: Optional[Dict[str, Any]] = {}

# ==================== REQUIREMENTS (Enhanced) ====================

class Requirement(BaseModel):
    id: Optional[int] = None
    project_id: Optional[int] = None
    name: str
    description: str
    category: str = "functional"
    priority: str = "Must Have"
    
    # NEW: Traceability to documents
    source_documents: List[int] = []  # Document IDs that support this requirement
    source_chunks: List[str] = []  # Specific chunk IDs for evidence
    
    # NEW: Regulatory/Compliance
    regulatory_mandate: bool = False
    regulatory_references: List[str] = []  # e.g., ["GDPR Art. 17", "HIPAA §164.312"]
    
    # Original fields
    source: str = "manual"
    created_at: str = datetime.now().isoformat()
    metadata: Optional[Dict[str, Any]] = {}

# ==================== STAKEHOLDER REQUIREMENTS (Vector-indexed) ====================

class StakeholderRequirement(BaseModel):
    """A stakeholder requirement stored in both JSON storage and ChromaDB."""
    id: Optional[int] = None
    project_id: int
    title: str
    description: str
    stakeholder_role: str  # "PM", "Engineering Lead", etc.
    stakeholder_id: Optional[int] = None
    category: str = "functional"
    priority: str = "Must Have"
    source: str = "manual"  # "manual", "bulk_import", "ai_extracted", "dashboard_chat"
    import_batch_id: Optional[str] = None
    regulatory_mandate: bool = False
    regulatory_references: List[str] = []
    created_at: str = datetime.now().isoformat()
    metadata: Optional[Dict[str, Any]] = {}

# ==================== USER STORIES (Enhanced) ====================

class UserStory(BaseModel):
    id: Optional[int] = None
    project_id: Optional[int] = None
    requirement_id: Optional[int] = None
    stakeholder_id: Optional[int] = None
    title: str
    description: str
    acceptance_criteria: Optional[str] = ""
    category: str = "functional"
    priority: Optional[str] = "Must Have"

    # NEW: Traceability
    source_documents: List[int] = []  # Documents that led to this story

    status: str = "ai_generated"  # pending_review, accepted, rejected, pm_validated, manual
    pm_validated: bool = False
    stakeholder_validated: bool = False
    created_at: str = datetime.now().isoformat()
    metadata: Optional[Dict[str, Any]] = {}

# ==================== STAKEHOLDER MANAGEMENT ====================

class Stakeholder(BaseModel):
    id: Optional[int] = None
    project_id: int
    name: str
    email: Optional[str] = None
    role: str  # "PM", "Engineering Lead", "Legal Counsel", "Compliance Officer"
    type: StakeholderType
    
    # Responsibilities & Concerns
    responsibilities: List[str] = []
    concerns: List[str] = []
    
    # Context filtering for RAG
    relevant_document_classifications: List[DocumentClassification] = []
    relevant_requirement_categories: List[str] = []
    risk_areas: List[str] = []  # "security", "compliance", "delivery"
    
    # Validation token for external stakeholders
    validation_token: Optional[str] = None
    
    metadata: Optional[Dict[str, Any]] = {}

# ==================== GAP ANALYSIS ====================

class Gap(BaseModel):
    gap_type: str  # "missing_document", "unclear_requirement", "regulatory_gap"
    description: str
    severity: str  # "critical", "high", "medium", "low"
    recommendation: str

class GapAnalysis(BaseModel):
    id: Optional[int] = None
    project_id: int
    analysis_date: str = datetime.now().isoformat()
    
    identified_gaps: List[Gap] = []
    missing_documents: List[str] = []  # Types of docs we expect but don't have
    unclear_areas: List[str] = []
    recommendations: List[str] = []
    
    # Based on uploaded documents vs project metadata
    coverage_score: float = 0.0  # 0-100%

# ==================== RISK MANAGEMENT ====================

class Risk(BaseModel):
    id: Optional[int] = None
    project_id: int
    requirement_id: Optional[int] = None
    document_id: Optional[int] = None
    
    risk_type: str  # "legal", "delivery", "security", "UX", "technical", "compliance"
    severity: str  # "critical", "high", "medium", "low"
    
    title: str
    description: str
    impact: str
    likelihood: str  # "very_high", "high", "medium", "low", "very_low"
    
    mitigation: Optional[str] = None
    owner: Optional[str] = None
    status: str = "identified"  # identified, mitigating, mitigated, accepted
    
    # Traceability
    source_document_chunks: List[str] = []  # Where was this risk identified?
    
    created_at: str = datetime.now().isoformat()

# ==================== ASSUMPTIONS REGISTER ====================

class Assumption(BaseModel):
    id: Optional[int] = None
    project_id: int
    requirement_id: Optional[int] = None
    
    assumption: str
    validation_status: str = "unvalidated"  # unvalidated, validated, invalidated
    impact_if_wrong: str
    validation_method: Optional[str] = None
    
    owner: Optional[str] = None
    created_at: str = datetime.now().isoformat()
    validated_at: Optional[str] = None

# ==================== TRACEABILITY ====================

class TraceabilityLink(BaseModel):
    id: Optional[int] = None
    project_id: int
    
    source_type: str  # "document", "requirement", "story", "risk", "assumption"
    source_id: int
    
    target_type: str
    target_id: int
    
    relationship: str  # "derived_from", "implements", "validates", "blocks", "supports"
    rationale: Optional[str] = None
    
    created_at: str = datetime.now().isoformat()

# ==================== RAG QUERY ====================

class RAGQuery(BaseModel):
    """Query for semantic search across documents"""
    query: str
    project_id: int
    
    # Filters
    document_classifications: List[DocumentClassification] = []
    stakeholder_filter: Optional[str] = None  # Filter by stakeholder concerns
    regulatory_only: bool = False
    
    # Results configuration
    top_k: int = 5  # Number of chunks to retrieve
    min_similarity: float = 0.7  # Minimum similarity threshold

class RAGResult(BaseModel):
    """Result from RAG query"""
    chunk_id: str
    document_id: int
    document_name: str
    content: str
    similarity_score: float
    
    # Context
    page_number: Optional[int] = None
    section: Optional[str] = None
    classification: str
    
    # Related entities
    related_requirements: List[int] = []
    related_risks: List[int] = []

class RetrievalResult(BaseModel):
    """Result from multi-collection retrieval."""
    chunk_id: str
    document_id: Optional[int] = None
    requirement_id: Optional[int] = None
    story_id: Optional[int] = None
    document_name: str
    content: str
    similarity_score: float
    source_collection: CollectionName
    # Context
    page_number: Optional[int] = None
    section: Optional[str] = None
    classification: Optional[str] = None
    stakeholder_role: Optional[str] = None
    category: Optional[str] = None

# ==================== BULK IMPORT ====================

class BulkImportItem(BaseModel):
    title: str
    description: str
    stakeholder_role: str
    category: str = "functional"
    priority: str = "Must Have"
    regulatory_mandate: bool = False
    regulatory_references: List[str] = []

class BulkImportResponse(BaseModel):
    success: bool
    import_batch_id: str
    imported_count: int
    errors: List[str] = []
    preview: List[StakeholderRequirement] = []

# ==================== ORIGINAL MODELS (KEPT FOR COMPATIBILITY) ====================

class ProcessMap(BaseModel):
    id: Optional[int] = None
    project_id: Optional[int] = None
    name: str
    type: str = "bpmn"
    steps: List[Dict[str, str]] = []
    created_at: str = datetime.now().isoformat()
    metadata: Optional[Dict[str, Any]] = {}

class ValidationToken(BaseModel):
    token: str
    project_id: int
    stakeholder_email: Optional[str] = None
    created_at: str = datetime.now().isoformat()
    expires_at: Optional[str] = None
    used: bool = False

class KnowledgeGraphNode(BaseModel):
    id: str
    type: str
    label: str
    data: Dict[str, Any] = {}

class KnowledgeGraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    metadata: Optional[Dict[str, Any]] = {}

class KnowledgeGraph(BaseModel):
    id: Optional[int] = None
    project_id: int
    nodes: List[KnowledgeGraphNode] = []
    edges: List[KnowledgeGraphEdge] = []
    created_at: str = datetime.now().isoformat()
    updated_at: str = datetime.now().isoformat()
    metadata: Optional[Dict[str, Any]] = {}

# Response models for AI services
class ClarifyingQuestion(BaseModel):
    requirement_id: int
    question: str
    category: str

class MoSCoWPriority(BaseModel):
    requirement_id: int
    priority: str
    reasoning: str

class RAGAnswerRequest(BaseModel):
    query: str
    project_id: int
    top_k: int = 20
    min_similarity: float = 0.5
    # optional filters
    document_classifications: List[DocumentClassification] = []

class RAGAnswerResponse(BaseModel):
    answer: List[str]
    acceptance_criteria: List[str]
    edge_cases: List[str]
    open_questions: List[str]
    citations_used: List[str]

# ==================== DASHBOARD MODELS ====================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class DashboardChatRequest(BaseModel):
    messages: List[ChatMessage]

class DashboardChatResponse(BaseModel):
    reply: str
    finished: bool
    draft_requirement: Optional[Dict[str, Any]] = None

class DashboardRequirementSubmission(BaseModel):
    title: str
    description: str
    category: str = "functional"
    priority: str = "Must Have"
    regulatory_mandate: bool = False
    regulatory_references: List[str] = []
