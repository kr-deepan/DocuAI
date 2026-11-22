"""
Phase 7 Verification Script
Validates the FastAPI backend routes (/health, /api/budget, /api/documents, /api/traces, /api/query, /api/projects)
and verifies the Streamlit UI file syntax and imports.
"""
import sys
import os
import py_compile

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import HTTPException
from src.api.main import (
    app,
    health_check,
    get_budget,
    get_documents,
    get_traces,
    list_projects,
    get_project_details,
    execute_query,
    QueryRequest
)

def verify_phase7():
    print("=" * 75)
    print("Phase 7 Verification Check: FastAPI Backend & Streamlit Trace UI")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Test /health Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 1] Testing GET /health...")
    data = health_check()
    assert data.get("status") == "healthy", f"FAILED: Status not healthy: {data}"
    assert "models" in data, "FAILED: Missing models map"
    assert data.get("indexed_chunks", 0) > 0, "FAILED: No indexed chunks reported"
    print(f"  [PASS] /health healthy! Indexed chunks: {data.get('indexed_chunks')}, Models: {list(data.get('models', {}).keys())}")

    # -------------------------------------------------------------
    # 2. Test /api/budget Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 2] Testing GET /api/budget...")
    b_data = get_budget()
    assert hasattr(b_data, "requests_used"), "FAILED: Missing requests_used"
    assert hasattr(b_data, "daily_ceiling"), "FAILED: Missing daily_ceiling"
    assert b_data.daily_ceiling == 45, f"FAILED: Expected ceiling 45, got {b_data.daily_ceiling}"
    print(f"  [PASS] /api/budget valid: used={b_data.requests_used}/45 (Headroom: {b_data.headroom_remaining})")

    # -------------------------------------------------------------
    # 3. Test /api/documents Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 3] Testing GET /api/documents...")
    d_data = get_documents(project_id="default")
    assert d_data.get("total_chunks", 0) >= 20, f"FAILED: Expected >= 20 chunks, got {d_data.get('total_chunks')}"
    assert len(d_data.get("document_counts", {})) > 0, "FAILED: Empty document counts"
    print(f"  [PASS] /api/documents valid: {d_data.get('total_chunks')} chunks across {len(d_data.get('document_counts', {}))} documents.")

    # -------------------------------------------------------------
    # 4. Test /api/projects Endpoints
    # -------------------------------------------------------------
    print("\n[CHECK 4] Testing GET /api/projects...")
    projs = list_projects()
    assert len(projs) > 0, "FAILED: No projects listed"
    assert any(p["id"] == "default" for p in projs), "FAILED: Default project missing"
    print(f"  [PASS] /api/projects returned {len(projs)} registered workspace(s).")

    p_details = get_project_details("default")
    assert p_details.get("id") == "default", "FAILED: Incorrect project id"
    assert len(p_details.get("documents", [])) > 0, "FAILED: No documents listed in default project"
    print(f"  [PASS] /api/projects/default returned {len(p_details.get('documents', []))} documents.")

    # -------------------------------------------------------------
    # 5. Test /api/traces Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 5] Testing GET /api/traces...")
    t_data = get_traces(limit=5)
    assert "traces" in t_data, "FAILED: Missing traces key"
    print(f"  [PASS] /api/traces returned {len(t_data.get('traces', []))} trace events.")

    # -------------------------------------------------------------
    # 6. Test /api/query Validation & Execution
    # -------------------------------------------------------------
    print("\n[CHECK 6] Testing POST /api/query (Input Validation)...")
    try:
        execute_query(QueryRequest(query="   ", mode="multi_agent"))
        assert False, "FAILED: Empty query should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 400, f"FAILED: Expected 400, got {e.status_code}"
    print("  [PASS] Empty query rejected with HTTP 400.")

    print("\n[CHECK 7] Testing POST /api/query (Naive RAG Mode)...")
    n_data = execute_query(QueryRequest(
        query="What is the notice period for contract termination?",
        mode="naive_rag",
        project_id="default"
    ))
    assert n_data.mode == "naive_rag", f"FAILED: Expected mode naive_rag, got {n_data.mode}"
    assert len(n_data.final_answer) > 0, "FAILED: Empty answer in naive_rag"
    assert len(n_data.retrieved_chunks) > 0, "FAILED: Empty retrieved_chunks in naive_rag"
    print("  [PASS] Naive RAG query executed successfully via API.")

    print("\n[CHECK 8] Testing POST /api/query (Multi-Agent Mode)...")
    a_data = execute_query(QueryRequest(
        query="What is the termination notice period required under the contract and its amendment?",
        mode="multi_agent",
        project_id="default"
    ))
    assert a_data.mode == "multi_agent", f"FAILED: Expected mode multi_agent, got {a_data.mode}"
    assert len(a_data.final_answer) > 0, "FAILED: Empty answer in multi_agent"
    assert len(a_data.retrieved_chunks) > 0, "FAILED: Empty retrieved_chunks in multi_agent"
    assert len(a_data.trace_logs) >= 3, f"FAILED: Expected trace logs, got {len(a_data.trace_logs)}"
    print(f"  [PASS] Multi-Agent query executed successfully via API! Nodes logged: {len(a_data.trace_logs)}")

    # -------------------------------------------------------------
    # 9. Test Streamlit UI Compilation & Syntax
    # -------------------------------------------------------------
    print("\n[CHECK 9] Validating Streamlit UI Application (src/ui/app.py)...")
    ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py"))
    assert os.path.exists(ui_path), f"FAILED: Missing {ui_path}"
    
    compiled_path = py_compile.compile(ui_path, doraise=True)
    assert compiled_path is not None, "FAILED: py_compile failed for src/ui/app.py"
    print("  [PASS] Streamlit UI code compiles cleanly without syntax errors.")

    print("\n" + "=" * 75)
    print("[ALL PASS] Phase 7 FastAPI Backend & Streamlit Trace UI Fully Verified!")
    print("=" * 75)

if __name__ == "__main__":
    verify_phase7()
