"""
Phase 5 Verification Script
Validates the Contradiction Agent node, cross-document conflict detection,
governing hierarchy resolution, and end-to-end 5-node LangGraph execution.
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.state import AgentState
from src.agents.contradiction import contradiction_node
from src.graph import run_agent_system

def verify_phase5():
    print("=" * 75)
    print("Phase 5 Verification Check: Contradiction Agent & 5-Node Graph")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Test Non-Contradictory Query (No False Positives)
    # -------------------------------------------------------------
    print("\n[CHECK 1] Testing Non-Contradictory Case (No False Positives)...")
    non_conflicting_chunks = [
        {
            "chunk_id": "master_service_agreement_v1_chunk_1",
            "text": "Provider agrees to deliver managed cloud infrastructure and document analytics software.",
            "metadata": {"source_doc": "master_service_agreement_v1.txt", "version": "v1", "section_header": "1. SCOPE"}
        }
    ]
    state_no_conflict: AgentState = {
        "original_query": "What services are covered?",
        "draft_answer": "Provider delivers managed cloud infrastructure [master_service_agreement_v1_chunk_1].",
        "retrieved_chunks": non_conflicting_chunks,
        "trace_logs": []
    }
    res_no_conflict = contradiction_node(state_no_conflict)
    assert len(res_no_conflict["contradictions_found"]) == 0, "FAILED: False positive contradiction detected!"
    print("  [PASS] Zero false positives detected on non-conflicting documents.")

    # -------------------------------------------------------------
    # 2. Test Contract vs. Amendment Contradiction (Notice Period & Payment)
    # -------------------------------------------------------------
    print("\n[CHECK 2] Testing Contract vs. Amendment Contradiction (Notice Period & Payment)...")
    msa_chunks = [
        {
            "chunk_id": "master_service_agreement_v1_chunk_4",
            "text": "Either party may terminate this Agreement without cause by providing at least thirty (30) days written notice.",
            "metadata": {"source_doc": "master_service_agreement_v1.txt", "version": "v1", "section_header": "8. TERM AND TERMINATION"}
        },
        {
            "chunk_id": "msa_amendment_v2_chunk_4",
            "text": "Section 8.1 of MSA v1 is hereby amended. Either party may terminate without cause by providing at least ninety (90) days written notice.",
            "metadata": {"source_doc": "msa_amendment_v2.txt", "version": "v2", "section_header": "8. AMENDED TERM AND TERMINATION"}
        },
        {
            "chunk_id": "master_service_agreement_v1_chunk_2",
            "text": "All invoices are due and payable within Net-30 days from invoice date.",
            "metadata": {"source_doc": "master_service_agreement_v1.txt", "version": "v1", "section_header": "4. PAYMENT & INVOICING"}
        },
        {
            "chunk_id": "msa_amendment_v2_chunk_2",
            "text": "All invoices are due and payable within Net-60 days from invoice date.",
            "metadata": {"source_doc": "msa_amendment_v2.txt", "version": "v2", "section_header": "4. AMENDED PAYMENT & INVOICING"}
        }
    ]

    state_msa_conflict: AgentState = {
        "original_query": "What is the termination notice period and payment terms?",
        "draft_answer": "Draft discusses 30 days [master_service_agreement_v1_chunk_4] and 90 days [msa_amendment_v2_chunk_4].",
        "retrieved_chunks": msa_chunks,
        "trace_logs": []
    }
    res_msa_conflict = contradiction_node(state_msa_conflict)
    conflicts = res_msa_conflict["contradictions_found"]
    
    assert len(conflicts) >= 1, "FAILED: Failed to detect MSA v1 vs v2 contradiction!"
    dims = [c["dimension"] for c in conflicts]
    print(f"  [PASS] Detected {len(conflicts)} conflict dimensions: {dims}")
    
    notice_conflict = next(c for c in conflicts if "Notice" in c["dimension"])
    assert notice_conflict["governing_document"] == "msa_amendment_v2.txt", "FAILED: Governing document resolution incorrect!"
    assert "Contradiction & Version Comparison Alert" in res_msa_conflict["final_answer"], "FAILED: Alert notice missing from final_answer!"
    print(f"  [PASS] Correct governing document identified: '{notice_conflict['governing_document']}'")
    print(f"  Precedence Rationale: {notice_conflict['resolution_rationale']}")

    # -------------------------------------------------------------
    # 3. Test General Policy vs. DPA Contradiction (Breach Notification)
    # -------------------------------------------------------------
    print("\n[CHECK 3] Testing Policy vs. Addendum Contradiction (Breach Notification Timeline)...")
    dpa_chunks = [
        {
            "chunk_id": "general_security_policy_chunk_2",
            "text": "Provider shall notify affected clients within seventy-two (72) hours of breach verification.",
            "metadata": {"source_doc": "general_security_policy.txt", "version": "2023", "section_header": "3. INCIDENT NOTIFICATION"}
        },
        {
            "chunk_id": "data_processing_addendum_2024_chunk_2",
            "text": "Notwithstanding 72-hour general policy windows, data breach notifications involving EU Personal Data must be communicated within twenty-four (24) hours.",
            "metadata": {"source_doc": "data_processing_addendum_2024.txt", "version": "2024", "section_header": "4. DATA BREACH MANAGEMENT"}
        }
    ]

    state_dpa_conflict: AgentState = {
        "original_query": "What is the breach notification window?",
        "draft_answer": "Policy specifies 72 hours [general_security_policy_chunk_2], while DPA specifies 24 hours [data_processing_addendum_2024_chunk_2].",
        "retrieved_chunks": dpa_chunks,
        "trace_logs": []
    }
    res_dpa_conflict = contradiction_node(state_dpa_conflict)
    dpa_conflicts = res_dpa_conflict["contradictions_found"]
    assert len(dpa_conflicts) >= 1, "FAILED: Failed to detect 72h vs 24h breach notification conflict!"
    assert dpa_conflicts[0]["governing_document"] == "data_processing_addendum_2024.txt", "FAILED: DPA not resolved as governing authority!"
    print(f"  [PASS] Detected breach notification timeline conflict (72h vs 24h).")
    print(f"  Governing Authority: '{dpa_conflicts[0]['governing_document']}'")

    # -------------------------------------------------------------
    # 4. Test Full 5-Node LangGraph Execution End-to-End
    # -------------------------------------------------------------
    print("\n[CHECK 4] Testing Full 5-Node LangGraph Pipeline Execution...")
    graph_query = "Compare the termination notice period and payment terms between the contract and amendment."
    final_state = run_agent_system(graph_query)

    agent_names = [t.get("agent_name") for t in final_state.get("trace_logs", [])]
    expected_nodes = ["orchestrator", "retriever", "synthesis", "critic", "contradiction"]
    for node in expected_nodes:
        assert node in agent_names, f"FAILED: Node '{node}' was not invoked in graph execution!"

    assert len(final_state.get("contradictions_found", [])) >= 1, "FAILED: No contradictions found in final state!"
    assert "Contradiction & Version Comparison Alert" in final_state.get("final_answer", ""), "FAILED: Final answer missing contradiction alert box!"

    print(f"  [PASS] Full 5-node graph executed successfully!")
    print(f"  Full Node Trace: {' -> '.join(agent_names)}")
    print(f"  Total Contradictions Detected: {len(final_state['contradictions_found'])}")
    print(f"\n  Final Answer with Contradiction Alert Box:\n  -----------------------------------------")
    print(final_state['final_answer'])

    print("\n" + "=" * 75)
    print("Phase 5 Verification: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    verify_phase5()
