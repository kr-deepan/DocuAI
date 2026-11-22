"""
Multi-Project Workspace Strict Isolation Verification Suite

Validates that:
1. New isolated project workspaces can be created and managed via DatabaseManager.
2. Documents can be ingested independently into separate project workspaces.
3. Chunks are strictly partitioned into distinct ChromaDB collections and BM25 indices.
4. Queries run on Project Alpha retrieve ZERO chunks from Project Beta or default corpus.
5. Queries run on Project Beta retrieve ZERO chunks from Project Alpha or default corpus.
6. The full 5-node agent pipeline respects project boundaries.
7. Project deletion cascades properly without affecting default or neighboring projects.
"""
import sys
import os
import time

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.database import db_manager
from src.vector_store import vector_store
from src.ingestion import ingest_project_document
from src.retrieval.hybrid_search import get_hybrid_retriever
from src.graph import run_agent_system

def run_multi_project_verification():
    print("=" * 70)
    print("DocuAI: Multi-Project Workspace Strict Isolation Verification")
    print("=" * 70)

    proj_alpha = "verify_proj_alpha"
    proj_beta = "verify_proj_beta"

    # Clean up any leftover test data first
    vector_store.delete_project_collection(proj_alpha)
    vector_store.delete_project_collection(proj_beta)
    db_manager.delete_project(proj_alpha)
    db_manager.delete_project(proj_beta)

    # -------------------------------------------------------------------------
    # Test 1: Project Creation
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Creating Isolated Projects in SQLite...")
    alpha_info = db_manager.create_project(
        project_id=proj_alpha,
        name="Project Alpha Corporation",
        description="High-security enterprise SLA and vendor contracts"
    )
    beta_info = db_manager.create_project(
        project_id=proj_beta,
        name="Project Beta Logistics",
        description="Agile supply chain agreements"
    )

    all_projs = db_manager.list_projects()
    proj_ids = [p["id"] for p in all_projs]
    assert proj_alpha in proj_ids, f"Project {proj_alpha} not listed in DB!"
    assert proj_beta in proj_ids, f"Project {proj_beta} not listed in DB!"
    assert "default" in proj_ids, "Default project missing from DB!"
    print(f"  ✅ Created '{proj_alpha}' and '{proj_beta}'. Total registered projects: {len(all_projs)}")

    # -------------------------------------------------------------------------
    # Test 2: Ingest Distinct Conflicting Documents into Alpha & Beta
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Ingesting Isolated Documents with Conflicting Terms...")
    doc_alpha_text = (
        "ALPHA CORPORATION MASTER AGREEMENT (v1.0)\n\n"
        "SECTION 1. TERMINATION\n"
        "Either party may terminate this agreement upon 180 days advance written notice. "
        "An early termination penalty fee of $50,000 shall immediately become due and payable.\n\n"
        "SECTION 2. PAYMENT TERMS\n"
        "All invoices are payable on Net-90 terms with a 5% monthly late fee compounding."
    )

    doc_beta_text = (
        "BETA LOGISTICS VENDOR AGREEMENT (v1.0)\n\n"
        "SECTION 1. TERMINATION\n"
        "Either party may terminate this agreement upon 15 days advance written notice. "
        "There is zero early termination penalty fee under any circumstance.\n\n"
        "SECTION 2. PAYMENT TERMS\n"
        "All invoices are payable on Net-15 terms with zero late penalty fees."
    )

    chunks_alpha = ingest_project_document(
        project_id=proj_alpha,
        filename="alpha_master_agreement.txt",
        content=doc_alpha_text,
        file_type="contract"
    )
    print(f"  Indexed {chunks_alpha} chunks into '{proj_alpha}'.")

    chunks_beta = ingest_project_document(
        project_id=proj_beta,
        filename="beta_vendor_agreement.txt",
        content=doc_beta_text,
        file_type="contract"
    )
    print(f"  Indexed {chunks_beta} chunks into '{proj_beta}'.")

    # Verify counts in ChromaDB and SQLite
    count_alpha = vector_store.get_count(proj_alpha)
    count_beta = vector_store.get_count(proj_beta)
    count_default = vector_store.get_count("default")

    print(f"  Chroma collection counts -> Alpha: {count_alpha}, Beta: {count_beta}, Default: {count_default}")
    assert count_alpha > 0, "Alpha collection is empty!"
    assert count_beta > 0, "Beta collection is empty!"
    assert count_default >= 21, f"Default collection was corrupted! Count: {count_default}"

    docs_alpha_db = db_manager.get_project_documents(proj_alpha)
    docs_beta_db = db_manager.get_project_documents(proj_beta)
    assert len(docs_alpha_db) == 1, "Alpha SQLite doc count mismatch!"
    assert len(docs_beta_db) == 1, "Beta SQLite doc count mismatch!"
    assert docs_alpha_db[0]["filename"] == "alpha_master_agreement.txt"
    assert docs_beta_db[0]["filename"] == "beta_vendor_agreement.txt"
    print("  ✅ Document records and chunk counts verified in SQLite and ChromaDB.")

    # -------------------------------------------------------------------------
    # Test 3: Strict Isolation Retrieval Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verifying Strict Hybrid Retrieval Isolation (Zero Cross-Leakage)...")
    
    # Query Alpha for termination
    retriever_alpha = get_hybrid_retriever(proj_alpha)
    results_alpha = retriever_alpha.search("termination notice period and penalty fee", top_k=5)
    
    print(f"  Alpha Query Results Count: {len(results_alpha)}")
    assert len(results_alpha) > 0, "Alpha search returned no results!"
    assert any("180 days" in r.get("text", "") for r in results_alpha), "Expected 180 days in Alpha retrieved chunks!"
    
    for r in results_alpha:
        src = r.get("metadata", {}).get("source_doc", "")
        pid = r.get("metadata", {}).get("project_id", "")
        text = r.get("text", "")
        print(f"    - Alpha Result: [{src}] proj={pid} text='{text[:60]}...'")
        assert "beta" not in src.lower(), f"DATA LEAKAGE: Beta doc '{src}' found in Alpha query!"
        assert "sample" not in src.lower(), f"DATA LEAKAGE: Default doc '{src}' found in Alpha query!"
        assert pid == proj_alpha, f"Project ID metadata mismatch in Alpha result: {pid}"

    # Query Beta for termination
    retriever_beta = get_hybrid_retriever(proj_beta)
    results_beta = retriever_beta.search("termination notice period and penalty fee", top_k=5)
    
    print(f"  Beta Query Results Count: {len(results_beta)}")
    assert len(results_beta) > 0, "Beta search returned no results!"
    assert any("15 days" in r.get("text", "") for r in results_beta), "Expected 15 days in Beta retrieved chunks!"

    for r in results_beta:
        src = r.get("metadata", {}).get("source_doc", "")
        pid = r.get("metadata", {}).get("project_id", "")
        text = r.get("text", "")
        print(f"    - Beta Result: [{src}] proj={pid} text='{text[:60]}...'")
        assert "alpha" not in src.lower(), f"DATA LEAKAGE: Alpha doc '{src}' found in Beta query!"
        assert "sample" not in src.lower(), f"DATA LEAKAGE: Default doc '{src}' found in Beta query!"
        assert pid == proj_beta, f"Project ID metadata mismatch in Beta result: {pid}"

    print("  ✅ Mathematical Isolation Confirmed: Zero cross-project leakage between Alpha and Beta.")

    # -------------------------------------------------------------------------
    # Test 4: End-to-End 5-Node Agent System Scoped to Project
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Executing Full 5-Node LangGraph State Graph on Project Alpha...")
    alpha_state = run_agent_system(
        query="What is the termination notice period and early exit fee?",
        project_id=proj_alpha
    )
    final_ans = alpha_state.get("final_answer", "")
    citations = alpha_state.get("citations", [])
    retrieved_alpha = alpha_state.get("retrieved_chunks", [])

    print(f"  Final Answer:\n{final_ans[:300]}...")
    print(f"  Retrieved Chunks in Agent: {len(retrieved_alpha)}")
    print(f"  Citations: {citations}")

    assert len(retrieved_alpha) > 0, "Agent pipeline retrieved 0 chunks for Alpha!"
    for r in retrieved_alpha:
        src = r.get("metadata", {}).get("source_doc", "")
        assert "beta" not in src.lower(), f"DATA BLEED: Agent retrieved Beta chunk '{src}' in Alpha run!"
        assert "sample" not in src.lower(), f"DATA BLEED: Agent retrieved Default chunk '{src}' in Alpha run!"

    assert "180" in final_ans or "180 days" in str(retrieved_alpha), "Expected 180 days referenced in Alpha pipeline!"
    assert "15 days" not in final_ans, f"DATA BLEED: Beta 15-day term found in Alpha answer! Got: {final_ans}"
    print("  ✅ End-to-end agent system correctly scoped to Project Alpha.")

    # -------------------------------------------------------------------------
    # Test 5: Project Deletion & Cleanup
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Verifying Project Deletion & Clean Purge...")
    del_alpha = db_manager.delete_project(proj_alpha)
    vector_store.delete_project_collection(proj_alpha)
    del_beta = db_manager.delete_project(proj_beta)
    vector_store.delete_project_collection(proj_beta)

    remaining_projs = [p["id"] for p in db_manager.list_projects()]
    assert proj_alpha not in remaining_projs, "Alpha still in DB after deletion!"
    assert proj_beta not in remaining_projs, "Beta still in DB after deletion!"
    assert "default" in remaining_projs, "Default project was deleted!"
    assert vector_store.get_count("default") >= 21, "Default collection chunks lost!"

    # Verify default cannot be deleted
    cannot_delete_default = db_manager.delete_project("default")
    assert not cannot_delete_default, "Default project should be protected from deletion!"

    print("  ✅ Projects cleanly deleted. Default project remains 100% intact.")

    print("\n" + "=" * 70)
    print("🎉 ALL MULTI-PROJECT ISOLATION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_multi_project_verification()
