"""
Phase 3 Verification Script
Validates the LangGraph 3-node agent state machine (Orchestrator + Retrieval + Synthesis),
Hybrid BM25 + Dense search with RRF, inline chunk citations, and trace logging.
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_search import hybrid_retriever
from src.agents.state import AgentState
from src.agents.orchestrator import orchestrator_node
from src.agents.retrieval import retrieval_node
from src.agents.synthesis import synthesis_node
from src.graph import run_agent_system, agent_app
from src.core.trace_logger import TRACE_LOG_PATH

def verify_phase3():
    print("=" * 75)
    print("Phase 3 Verification Check: LangGraph Multi-Agent System (3 Nodes)")
    print("=" * 75)

    # 1. Test Hybrid Search (Dense + BM25 + RRF)
    print("\n[CHECK 1] Testing Hybrid Retriever (Dense + BM25 RRF Fusion)...")
    test_query = "termination notice period thirty days ninety days"
    hybrid_results = hybrid_retriever.search(test_query, top_k=4)
    
    assert len(hybrid_results) > 0, "FAILED: Hybrid retriever returned 0 chunks!"
    assert "rrf_score" in hybrid_results[0], "FAILED: RRF score missing from hybrid results!"
    print(f"  [PASS] Retrieved {len(hybrid_results)} chunks via Hybrid RRF search.")
    for idx, c in enumerate(hybrid_results, 1):
        meta = c.get("metadata", {})
        print(f"    [{idx}] {c['chunk_id']} | RRF Score: {c['rrf_score']:.4f} | Dense Rank: {c.get('dense_rank', 'N/A')} | BM25 Rank: {c.get('bm25_rank', 'N/A')}")

    # 2. Test Orchestrator Node Decomposition
    print("\n[CHECK 2] Testing Orchestrator Node Query Decomposition...")
    comp_query = "Compare payment terms and termination notice under MSA v1 and Amendment v2"
    orch_state: AgentState = {
        "original_query": comp_query,
        "trace_logs": []
    }
    orch_res = orchestrator_node(orch_state)
    assert orch_res["needs_retrieval"] is True, "FAILED: Orchestrator failed to set needs_retrieval=True!"
    assert len(orch_res["sub_queries"]) >= 1, "FAILED: Sub-queries list is empty!"
    print(f"  [PASS] Orchestrator generated {len(orch_res['sub_queries'])} sub-queries:")
    for sq in orch_res["sub_queries"]:
        print(f"    - '{sq}'")

    # 3. Test End-to-End LangGraph Execution
    print("\n[CHECK 3] Testing Compiled LangGraph Agent Pipeline End-to-End...")
    graph_query = "What is the termination notice period and payment terms under MSA v1 and Amendment v2?"
    final_state = run_agent_system(graph_query)

    # Validate state transitions
    assert final_state.get("final_answer"), "FAILED: Final answer is missing or empty!"
    assert len(final_state.get("retrieved_chunks", [])) > 0, "FAILED: No chunks in final state!"
    assert len(final_state.get("citations", [])) > 0, "FAILED: No inline chunk citations produced!"
    assert len(final_state.get("trace_logs", [])) >= 3, "FAILED: Expected at least 3 node trace steps!"

    print(f"  [PASS] Graph executed successfully.")
    print(f"  Total Nodes in Trace: {len(final_state['trace_logs'])}")
    for t in final_state["trace_logs"]:
        print(f"    -> Agent: {t['agent_name']} ({t['latency_ms']}ms | {t['tokens']} tokens)")

    print(f"\n  Final Answer Preview:\n  ---------------------\n  {final_state['final_answer'][:250]}...")
    print(f"\n  Extracted Chunk Citations: {final_state['citations']}")

    # 4. Verify Persistent Trace Log File
    print("\n[CHECK 4] Verifying JSON Lines Trace Log File on Disk...")
    assert os.path.exists(TRACE_LOG_PATH), f"FAILED: Trace log file not found at {TRACE_LOG_PATH}"
    with open(TRACE_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 0, "FAILED: Trace log file is empty!"
    print(f"  [PASS] Trace log file verified with {len(lines)} total recorded transitions.")

    print("\n" + "=" * 75)
    print("Phase 3 Verification: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    verify_phase3()
