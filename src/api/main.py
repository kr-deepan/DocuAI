"""
FastAPI Backend Application

Role in the System:
-------------------
Provides an asynchronous REST API interface exposing the Contradiction-Aware
Multi-Agent LangGraph state machine, the Naive RAG baseline, and multi-project isolated workspaces.

Endpoints:
----------
- GET    /health                               : Service health, model assignments, and vector store status
- GET    /api/projects                         : List all isolated project workspaces with doc/chunk counts
- POST   /api/projects                         : Create a new isolated project workspace
- GET    /api/projects/{project_id}            : Retrieve project details and list of uploaded documents
- DELETE /api/projects/{project_id}            : Delete a project, its documents, and its vector collection
- POST   /api/projects/{project_id}/documents  : Ingest a document via raw text content into the project
- POST   /api/projects/{project_id}/upload     : Upload and ingest a file (.pdf, .txt, .png) into the project
- POST   /api/query                            : Executes a query scoped to a project via Multi-Agent or Naive RAG
- GET    /api/budget                           : Live daily request quota metrics (used / 45 ceiling)
- GET    /api/traces                           : Recent JSON lines execution traces
- GET    /api/documents                        : Ingested corpus documents and chunk counts for a project
"""
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    ORCHESTRATOR_MODEL,
    RETRIEVAL_MODEL,
    CRITIC_MODEL,
    SYNTHESIS_MODEL,
    EMBEDDING_MODEL,
    DAILY_REQUEST_BUDGET
)
from src.core.budget import budget_guard
from src.core.trace_logger import trace_logger
from src.core.database import db_manager
from src.vector_store import vector_store
from src.ingestion import ingest_project_document, ingest_project_file_bytes
from src.retrieval.hybrid_search import get_hybrid_retriever
from src.graph import run_agent_system
from src.baselines.naive_rag import naive_rag
from src.agents.synthesis import extract_citations

app = FastAPI(
    title="DocuAI: Contradiction-Aware Multi-Agent Document Intelligence API",
    description="Enterprise Multi-Agent Document Intelligence API with LangGraph, OpenRouter, SQLite, and ChromaDB",
    version="2.0.0"
)

# Enable CORS for local web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class ProjectCreateRequest(BaseModel):
    id: str = Field(..., description="Unique alphanumeric identifier (e.g. 'healthcare_nda')")
    name: str = Field(..., description="Human-readable project name")
    description: Optional[str] = Field("", description="Optional workspace description")

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    created_at: Optional[str] = None
    document_count: int = 0
    total_chunks: int = 0

class DocumentUploadRequest(BaseModel):
    filename: str = Field(..., description="Document filename (e.g. 'vendor_agreement_v2.txt')")
    content: str = Field(..., description="Raw text content of the document")
    file_type: Optional[str] = Field("txt", description="Document type: 'txt', 'pdf', 'contract', etc.")

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to analyze against the document corpus")
    project_id: str = Field("default", description="Project workspace ID to query against (strict isolation)")
    mode: str = Field("multi_agent", description="'multi_agent' or 'naive_rag'")

class QueryResponse(BaseModel):
    query: str
    project_id: str
    mode: str
    final_answer: str
    citations: List[str]
    contradictions_found: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    trace_logs: List[Dict[str, Any]]
    verification_passed: bool
    verification_attempts: int

class BudgetResponse(BaseModel):
    requests_used: int
    daily_ceiling: int
    pool_limit: int
    headroom_remaining: int

# -----------------------------------------------------------------------------
# System & Budget Routes
# -----------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Returns system status, active models, vector store status, and project counts."""
    projects = db_manager.list_projects()
    default_cnt = vector_store.get_count("default")
    return {
        "status": "healthy",
        "models": {
            "orchestrator": ORCHESTRATOR_MODEL,
            "retrieval": RETRIEVAL_MODEL,
            "critic": CRITIC_MODEL,
            "synthesis": SYNTHESIS_MODEL,
            "embedding": EMBEDDING_MODEL
        },
        "total_projects": len(projects),
        "indexed_chunks": default_cnt,
        "default_chunks": default_cnt,
        "budget_used": budget_guard.get_usage(),
        "budget_ceiling": DAILY_REQUEST_BUDGET
    }

@app.get("/api/budget", response_model=BudgetResponse, tags=["Budget"])
def get_budget():
    """Returns current daily request budget quota metrics."""
    used = budget_guard.get_usage()
    ceiling = DAILY_REQUEST_BUDGET
    return BudgetResponse(
        requests_used=used,
        daily_ceiling=ceiling,
        pool_limit=50,
        headroom_remaining=max(0, ceiling - used)
    )

@app.get("/api/traces", tags=["Observability"])
def get_traces(limit: int = 20):
    """Returns the latest agent execution traces from disk."""
    return {"traces": trace_logger.get_recent_traces(limit=limit)}

# -----------------------------------------------------------------------------
# Multi-Project Workspace Management Routes
# -----------------------------------------------------------------------------

@app.get("/api/projects", response_model=List[ProjectResponse], tags=["Projects"])
def list_projects():
    """Returns all projects with their document and chunk counts."""
    return db_manager.list_projects()

@app.post("/api/projects", response_model=ProjectResponse, tags=["Projects"])
def create_project(req: ProjectCreateRequest):
    """Creates a new isolated project workspace."""
    clean_id = req.id.strip().lower().replace(" ", "_")
    if not clean_id:
        raise HTTPException(status_code=400, detail="Project ID cannot be empty.")
    
    existing = db_manager.get_project(clean_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Project '{clean_id}' already exists.")
    
    proj = db_manager.create_project(
        project_id=clean_id,
        name=req.name,
        description=req.description or ""
    )
    return {**proj, "document_count": 0, "total_chunks": 0}

@app.get("/api/projects/{project_id}", tags=["Projects"])
def get_project_details(project_id: str):
    """Retrieves project details and its list of ingested documents."""
    proj = db_manager.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    
    documents = db_manager.get_project_documents(project_id)
    total_chunks = vector_store.get_count(project_id)
    return {
        **proj,
        "total_chunks": total_chunks,
        "documents": documents
    }

@app.delete("/api/projects/{project_id}", tags=["Projects"])
def delete_project(project_id: str):
    """Deletes a project, cascading to SQLite documents and purging its ChromaDB collection."""
    if project_id == "default":
        raise HTTPException(status_code=403, detail="The default project workspace cannot be deleted.")
    
    proj = db_manager.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    
    # Clean vector store collection
    vector_store.delete_project_collection(project_id)
    # Delete from database
    success = db_manager.delete_project(project_id)
    return {"status": "deleted", "project_id": project_id, "success": success}

# -----------------------------------------------------------------------------
# Document Ingestion Routes (Project-Scoped)
# -----------------------------------------------------------------------------

@app.post("/api/projects/{project_id}/documents", tags=["Documents"])
def add_project_document(project_id: str, doc_req: DocumentUploadRequest):
    """Ingests a text document into a specific isolated project workspace."""
    proj = db_manager.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    
    chunk_count = ingest_project_document(
        project_id=project_id,
        filename=doc_req.filename,
        content=doc_req.content,
        file_type=doc_req.file_type or "txt"
    )
    return {
        "status": "success",
        "project_id": project_id,
        "filename": doc_req.filename,
        "chunks_indexed": chunk_count,
        "total_project_chunks": vector_store.get_count(project_id)
    }

@app.post("/api/projects/{project_id}/upload", tags=["Documents"])
async def upload_project_file(project_id: str, file: UploadFile = File(...)):
    """Uploads a file (.pdf, .txt, .png, etc.), extracts text, chunks, and indexes it into the project workspace."""
    proj = db_manager.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    try:
        chunk_count = ingest_project_file_bytes(
            project_id=project_id,
            file_bytes=file_bytes,
            filename=file.filename
        )
        return {
            "status": "success",
            "project_id": project_id,
            "filename": file.filename,
            "chunks_indexed": chunk_count,
            "total_project_chunks": vector_store.get_count(project_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/api/documents", tags=["Corpus"])
def get_documents(project_id: str = Query("default", description="Project workspace identifier")):
    """Returns list of unique documents and chunk count stored for a specific project."""
    stored = vector_store.get_all_chunks(project_id)
    docs_summary = {}
    if stored and stored.get("metadatas"):
        for meta in stored["metadatas"]:
            doc_name = meta.get("source_doc", "unknown")
            docs_summary[doc_name] = docs_summary.get(doc_name, 0) + 1

    return {
        "project_id": project_id,
        "total_chunks": vector_store.get_count(project_id),
        "document_counts": docs_summary
    }

# -----------------------------------------------------------------------------
# Query Execution Route (Project-Scoped)
# -----------------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse, tags=["Query Execution"])
def execute_query(req: QueryRequest):
    """
    Executes a query through either the full 5-node Multi-Agent system
    or the Naive Single-Shot RAG baseline, scoped strictly to the specified project.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    target_project = req.project_id or "default"

    if req.mode == "naive_rag":
        # Execute Naive Baseline with project isolation
        res = naive_rag.query(req.query, project_id=target_project)
        answer = res.get("answer", "")
        cits = extract_citations(answer)
        ret_chunks = res.get("retrieved_chunks", [])
        
        dummy_trace = [{
            "agent_name": "naive_rag_baseline",
            "input_summary": req.query,
            "output_summary": {"answer_length": len(answer)},
            "tokens": res.get("usage", {}).get("total_tokens", 0),
            "latency_ms": res.get("latency_ms", 0),
            "metadata": {"model": res.get("model_used"), "project_id": target_project}
        }]

        return QueryResponse(
            query=req.query,
            project_id=target_project,
            mode="naive_rag",
            final_answer=answer,
            citations=cits,
            contradictions_found=[],
            retrieved_chunks=ret_chunks,
            trace_logs=dummy_trace,
            verification_passed=False,
            verification_attempts=1
        )

    else:
        # Execute Full 5-Node Multi-Agent Graph with project isolation
        state = run_agent_system(req.query, project_id=target_project)
        return QueryResponse(
            query=req.query,
            project_id=target_project,
            mode="multi_agent",
            final_answer=state.get("final_answer", ""),
            citations=state.get("citations", []),
            contradictions_found=state.get("contradictions_found", []),
            retrieved_chunks=state.get("retrieved_chunks", []),
            trace_logs=state.get("trace_logs", []),
            verification_passed=state.get("verification_passed", True),
            verification_attempts=state.get("verification_attempts", 1)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
