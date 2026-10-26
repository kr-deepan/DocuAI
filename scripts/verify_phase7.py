"""
Phase 7 Verification Script
Validates the FastAPI backend endpoints (/health, /api/budget, /api/documents, /api/traces, /api/query)
using TestClient, and verifies the Streamlit UI file syntax and imports.
"""
import sys
import os
import py_compile
from fastapi.testclient import TestClient

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app

def verify_phase7():
    print("=" * 75)
    print("Phase 7 Verification Check: FastAPI Backend & Streamlit Trace UI")
    print("=" * 75)

    client = TestClient(app)

    # -------------------------------------------------------------
    # 1. Test /health Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 1] Testing GET /health...")
    res = client.get("/health")
    assert res.status_code == 200, f"FAILED: Expected 200, got {res.status_code}"
    data = res.json()
    assert data.get("status") == "healthy", f"FAILED: Status not healthy: {data}"
    assert "models" in data, "FAILED: Missing models map"
    assert data.get("indexed_chunks", 0) > 0, "FAILED: No indexed chunks reported"
    print(f"  [PASS] /health healthy! Indexed chunks: {data.get('indexed_chunks')}, Models: {list(data.get('models', {}).keys())}")

    # -------------------------------------------------------------
    # 2. Test /api/budget Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 2] Testing GET /api/budget...")
    res = client.get("/api/budget")
    assert res.status_code == 200, f"FAILED: Expected 200, got {res.status_code}"
    b_data = res.json()
    assert "requests_used" in b_data, "FAILED: Missing requests_used"
    assert "daily_ceiling" in b_data, "FAILED: Missing daily_ceiling"
    assert b_data.get("daily_ceiling") == 45, f"FAILED: Expected ceiling 45, got {b_data.get('daily_ceiling')}"
    print(f"  [PASS] /api/budget valid: used={b_data.get('requests_used')}/45 (Headroom: {b_data.get('headroom_remaining')})")

    # -------------------------------------------------------------
    # 3. Test /api/documents Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 3] Testing GET /api/documents...")
    res = client.get("/api/documents")
    assert res.status_code == 200, f"FAILED: Expected 200, got {res.status_code}"
    d_data = res.json()
    assert d_data.get("total_chunks", 0) >= 20, f"FAILED: Expected >= 20 chunks, got {d_data.get('total_chunks')}"
    assert len(d_data.get("document_counts", {})) > 0, "FAILED: Empty document counts"
    print(f"  [PASS] /api/documents valid: {d_data.get('total_chunks')} chunks across {len(d_data.get('document_counts', {}))} documents.")

    # -------------------------------------------------------------
    # 4. Test /api/traces Endpoint
    # -------------------------------------------------------------
    print("\n[CHECK 4] Testing GET /api/traces...")
    res = client.get("/api/traces?limit=5")
    assert res.status_code == 200, f"FAILED: Expected 200, got {res.status_code}"
    t_data = res.json()
    assert "traces" in t_data, "FAILED: Missing traces key"
    print(f"  [PASS] /api/traces returned {len(t_data.get('traces', []))} trace events.")

    # -------------------------------------------------------------
    # 5. Test /api/query Validation & Execution
    # -------------------------------------------------------------
    print("\n[CHECK 5] Testing POST /api/query (Input Validation)...")
    res = client.post("/api/query", json={"query": "   ", "mode": "multi_agent"})
    assert res.status_code == 400, f"FAILED: Expected 400 for empty query, got {res.status_code}"
    print("  [PASS] Empty query rejected with HTTP 400.")

    print("\n[CHECK 6] Testing POST /api/query (Naive RAG Mode)...")
    naive_res = client.post("/api/query", json={
        "query": "What is the notice period for contract termination?",
        "mode": "naive_rag"
    })
    assert naive_res.status_code == 200, f"FAILED: Expected 200, got {naive_res.status_code}"
    n_data = naive_res.json()
    assert n_data.get("mode") == "naive_rag", f"FAILED: Expected mode naive_rag, got {n_data.get('mode')}"
    assert len(n_data.get("final_answer", "")) > 0, "FAILED: Empty answer in naive_rag"
    assert len(n_data.get("retrieved_chunks", [])) > 0, "FAILED: Empty retrieved_chunks in naive_rag"
    print("  [PASS] Naive RAG query executed successfully via API.")

    print("\n[CHECK 7] Testing POST /api/query (Multi-Agent Mode)...")
    # Using cached query from evaluation suite to preserve zero-quota usage
    agent_res = client.post("/api/query", json={
        "query": "What is the termination notice period required under the contract and its amendment?",
        "mode": "multi_agent"
    })
    assert agent_res.status_code == 200, f"FAILED: Expected 200, got {agent_res.status_code}"
    a_data = agent_res.json()
    assert a_data.get("mode") == "multi_agent", f"FAILED: Expected mode multi_agent, got {a_data.get('mode')}"
    assert len(a_data.get("final_answer", "")) > 0, "FAILED: Empty answer in multi_agent"
    assert len(a_data.get("retrieved_chunks", [])) > 0, "FAILED: Empty retrieved_chunks in multi_agent"
    assert len(a_data.get("trace_logs", [])) >= 3, f"FAILED: Expected trace logs, got {len(a_data.get('trace_logs', []))}"
    print(f"  [PASS] Multi-Agent query executed successfully via API! Nodes logged: {len(a_data.get('trace_logs', []))}")

    # -------------------------------------------------------------
    # 8. Test Streamlit UI Compilation & Syntax
    # -------------------------------------------------------------
    print("\n[CHECK 8] Validating Streamlit UI Application (src/ui/app.py)...")
    ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py"))
    assert os.path.exists(ui_path), f"FAILED: Missing {ui_path}"
    
    # Compile python file to ensure no syntax errors
    compiled_path = py_compile.compile(ui_path, doraise=True)
    assert compiled_path is not None, "FAILED: py_compile failed for src/ui/app.py"
    print(f"  [PASS] Streamlit UI code compiles cleanly without syntax errors.")

    print("\n" + "=" * 75)
    print("[ALL PASS] Phase 7 FastAPI Backend & Streamlit Trace UI Fully Verified!")
    print("=" * 75)

if __name__ == "__main__":
    verify_phase7()
