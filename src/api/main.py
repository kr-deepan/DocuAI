"""
FastAPI Backend Application

Role in the System:
-------------------
Provides an asynchronous REST API interface exposing the Contradiction-Aware
Multi-Agent LangGraph state machine, the Naive RAG baseline, and observability traces.

Endpoints:
----------
- GET  /health          : Service health, model assignments, and vector store status
- POST /api/query       : Executes a query through either Multi-Agent or Naive RAG
- GET  /api/budget      : Live daily request quota metrics (used / 45 ceiling)
- GET  /api/traces      : Recent JSON lines execution traces
- GET  /api/documents   : Ingested corpus documents and chunk counts

Why a separate FastAPI backend:
-------------------------------
In production, multi-agent systems must be decoupled from UI presentations.
An API layer enables integration with diverse clients (web dashboards, Slack bots,
enterprise microservices) while providing centralized request budgeting and trace logging.
"""
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
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
from src.vector_store import vector_store
from src.graph import run_agent_system
from src.baselines.naive_rag import naive_rag
from src.agents.synthesis import extract_citations

app = FastAPI(
    title="Contradiction-Aware Document Intelligence API",
    description="Multi-Agent Document Intelligence API with LangGraph, OpenRouter, and ChromaDB",
    version="1.0.0"
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

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to analyze against the document corpus")
    mode: str = Field("multi_agent", description="'multi_agent' or 'naive_rag'")

class QueryResponse(BaseModel):
    query: str
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
# API Routes
# -----------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Returns system status, active models, and collection size."""
    return {
        "status": "healthy",
        "models": {
            "orchestrator": ORCHESTRATOR_MODEL,
            "retrieval": RETRIEVAL_MODEL,
            "critic": CRITIC_MODEL,
            "synthesis": SYNTHESIS_MODEL,
            "embedding": EMBEDDING_MODEL
        },
        "indexed_chunks": vector_store.get_count(),
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

@app.get("/api/documents", tags=["Corpus"])
def get_documents():
    """Returns list of unique documents and chunk count stored in ChromaDB."""
    stored = vector_store.collection.get()
    docs_summary = {}
    if stored and stored.get("metadatas"):
        for meta in stored["metadatas"]:
            doc_name = meta.get("source_doc", "unknown")
            doc_type = meta.get("doc_type", "general")
            docs_summary[doc_name] = docs_summary.get(doc_name, 0) + 1

    return {
        "total_chunks": vector_store.get_count(),
        "document_counts": docs_summary
    }

@app.post("/api/query", response_model=QueryResponse, tags=["Query Execution"])
def execute_query(req: QueryRequest):
    """
    Executes a query through either the full 5-node Multi-Agent system
    or the Naive Single-Shot RAG baseline.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    if req.mode == "naive_rag":
        # Execute Naive Baseline
        res = naive_rag.query(req.query)
        answer = res.get("answer", "")
        cits = extract_citations(answer)
        ret_chunks = res.get("retrieved_chunks", [])
        
        # Format dummy trace step for baseline
        dummy_trace = [{
            "agent_name": "naive_rag_baseline",
            "input_summary": req.query,
            "output_summary": {"answer_length": len(answer)},
            "tokens": res.get("usage", {}).get("total_tokens", 0),
            "latency_ms": res.get("latency_ms", 0),
            "metadata": {"model": res.get("model_used")}
        }]

        return QueryResponse(
            query=req.query,
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
        # Execute Full 5-Node Multi-Agent Graph
        state = run_agent_system(req.query)
        return QueryResponse(
            query=req.query,
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
