"""
Phase 4 Verification Script
Validates the Verification Critic Agent, programmatic citation checking,
hallucination detection, and the LangGraph self-correction loop.
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.state import AgentState
from src.agents.critic import critic_node, verify_citations_programmatically
from src.graph import run_agent_system

def verify_phase4():
    print("=" * 75)
    print("Phase 4 Verification Check: Verification Critic & Self-Correction Loop")
    print("=" * 75)

    sample_chunks = [
        {
            "chunk_id": "master_service_agreement_v1_chunk_4",
            "text": "Either party may terminate this Agreement without cause by providing at least thirty (30) days written notice.",
            "metadata": {"source_doc": "master_service_agreement_v1.txt", "section_header": "8. TERM AND TERMINATION"}
        },
        {
            "chunk_id": "msa_amendment_v2_chunk_4",
            "text": "Either party may terminate this Agreement without cause by providing at least ninety (90) days written notice to the other party.",
            "metadata": {"source_doc": "msa_amendment_v2.txt", "section_header": "8. AMENDED TERM AND TERMINATION"}
        }
    ]

    # -------------------------------------------------------------
    # 1. Test Programmatic Citation Validation
    # -------------------------------------------------------------
    print("\n[CHECK 1] Testing Programmatic Citation Validation...")
    
    # Case A: Valid citations
    valid_draft = "Under MSA v1 notice is 30 days [master_service_agreement_v1_chunk_4], while Amendment v2 requires 90 days [msa_amendment_v2_chunk_4]."
    is_valid, val_cits, hal_cits = verify_citations_programmatically(valid_draft, sample_chunks)
    assert is_valid is True, "FAILED: Valid citations were incorrectly flagged as invalid!"
    assert len(hal_cits) == 0, "FAILED: False positive hallucination detected!"
    print(f"  [PASS] Valid citations correctly verified: {val_cits}")

    # Case B: Hallucinated / Fabricated Chunk ID
    fake_draft = "Termination requires 180 days notice under the secret addendum [fabricated_doc_v3_chunk_99]."
    is_valid_fake, val_fake, hal_fake = verify_citations_programmatically(fake_draft, sample_chunks)
    assert is_valid_fake is False, "FAILED: Hallucinated chunk ID was not caught!"
    assert "fabricated_doc_v3_chunk_99" in hal_fake, "FAILED: Hallucinated ID not identified!"
    print(f"  [PASS] Programmatic check successfully caught fake citation: {hal_fake}")

    # -------------------------------------------------------------
    # 2. Test Critic Node Grounding Check on Grounded vs Unsupported
    # -------------------------------------------------------------
    print("\n[CHECK 2] Testing Critic Node Verification Logic...")

    # Grounded state
    grounded_state: AgentState = {
        "original_query": "What is the notice period?",
        "draft_answer": valid_draft,
        "retrieved_chunks": sample_chunks,
        "verification_attempts": 0,
        "needs_retrieval": True,
        "trace_logs": []
    }
    critic_res_grounded = critic_node(grounded_state)
    assert critic_res_grounded["verification_passed"] is True, "FAILED: Grounded answer failed critic check!"
    print(f"  [PASS] Grounded draft passed critic check on attempt 1.")

    # Unsupported / Fake citation state
    unsupported_state: AgentState = {
        "original_query": "What is the notice period?",
        "draft_answer": fake_draft,
        "retrieved_chunks": sample_chunks,
        "verification_attempts": 0,
        "needs_retrieval": True,
        "trace_logs": []
    }
    critic_res_unsupported = critic_node(unsupported_state)
    assert critic_res_unsupported["verification_passed"] is False, "FAILED: Unsupported draft passed critic check!"
    print(f"  [PASS] Unsupported draft rejected by critic. Feedback: {critic_res_unsupported['critic_feedback']}")

    # -------------------------------------------------------------
    # 3. Test Loop Guard: Attempt Capping
    # -------------------------------------------------------------
    print("\n[CHECK 3] Testing Loop Guard (Preventing Infinite Loops & Budget Drain)...")
    capped_state: AgentState = {
        "original_query": "What is the notice period?",
        "draft_answer": fake_draft,
        "retrieved_chunks": sample_chunks,
        "verification_attempts": 1,  # Second attempt
        "needs_retrieval": True,
        "trace_logs": []
    }
    critic_res_capped = critic_node(capped_state)
    assert critic_res_capped["verification_passed"] is True, "FAILED: Loop guard did not force termination at attempt 2!"
    assert "Grounding Verification Notice" in critic_res_capped["final_answer"], "FAILED: Disclaimer not appended to final answer!"
    print(f"  [PASS] Loop guard forced safe completion at attempt 2 with disclaimer appended.")

    # -------------------------------------------------------------
    # 4. Test End-to-End LangGraph Multi-Agent Execution with Critic
    # -------------------------------------------------------------
    print("\n[CHECK 4] Testing End-to-End LangGraph Pipeline with Critic Node...")
    graph_query = "What is the termination notice period under the contract and its amendment?"
    final_state = run_agent_system(graph_query)

    agent_names = [t.get("agent_name") for t in final_state.get("trace_logs", [])]
    assert "critic" in agent_names, "FAILED: Critic node was not invoked during graph execution!"
    assert final_state.get("verification_passed") is True, "FAILED: Final state verification_passed is not True!"
    assert final_state.get("final_answer"), "FAILED: Final answer is missing!"

    print(f"  [PASS] Graph executed with Critic node in active state machine.")
    print(f"  Trace Path: {' -> '.join(agent_names)}")
    print(f"  Critic Feedback: {final_state.get('critic_feedback')}")
    print(f"  Verification Attempts: {final_state.get('verification_attempts')}")
    print(f"\n  Final Answer Preview:\n  ---------------------\n  {final_state['final_answer'][:250]}...")

    print("\n" + "=" * 75)
    print("Phase 4 Verification: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    verify_phase4()
