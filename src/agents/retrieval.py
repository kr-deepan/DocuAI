"""
Retrieval Agent Node

Role in the System:
-------------------
The Retrieval Agent executes targeted search queries against the hybrid retrieval
engine (Dense + BM25 with RRF). It processes all sub-queries produced by the
Orchestrator, merges the retrieved candidate pools, deduplicates identical chunks,
and ranks them.

Inputs & Outputs:
-----------------
- Input: `state["sub_queries"]`, `state["retrieval_filters"]`
- Output updates to `AgentState`:
    - `retrieved_chunks`: List of deduplicated, ranked evidence chunks
    - `trace_logs`: Appended trace entry

Why a separate Retrieval Agent rather than embedding inside the Orchestrator or Synthesis:
-----------------------------------------------------------------------------------------
1. Modularity & Independent Scaling: Retrieval strategies (switching between vector DBs,
   adjusting RRF weights, adding cross-encoder rerankers) evolve independently of prompt logic.
2. Self-Correction Loop (Phase 4): When the downstream Verification Critic flags
   insufficient evidence, a conditional edge loops directly back to this Retrieval Agent
   with refined queries, without re-running the Orchestrator.
"""
import time
from typing import Dict, Any, List

from src.agents.state import AgentState
from src.retrieval.hybrid_search import hybrid_retriever
from src.core.trace_logger import trace_logger

class RetrievalNode:
    def __init__(self, top_k_per_query: int = 4, max_total_chunks: int = 6):
        self.top_k_per_query = top_k_per_query
        self.max_total_chunks = max_total_chunks

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        start_time = time.time()
        sub_queries = state.get("sub_queries") or [state.get("original_query", "")]
        where_filter = state.get("retrieval_filters")

        seen_chunks: Dict[str, Dict[str, Any]] = {}

        # Search hybrid retriever for each sub-query
        for query_str in sub_queries:
            results = hybrid_retriever.search(
                query=query_str,
                top_k=self.top_k_per_query,
                where_filter=where_filter
            )
            for chunk in results:
                cid = chunk["chunk_id"]
                # If already seen, keep highest RRF score
                if cid in seen_chunks:
                    if chunk.get("rrf_score", 0) > seen_chunks[cid].get("rrf_score", 0):
                        seen_chunks[cid] = chunk
                else:
                    seen_chunks[cid] = chunk

        # Sort combined chunks by RRF score descending
        merged_chunks = list(seen_chunks.values())
        merged_chunks.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        final_chunks = merged_chunks[:self.max_total_chunks]

        latency_ms = int((time.time() - start_time) * 1000)

        chunk_summaries = [
            f"{c['chunk_id']} ({c.get('metadata', {}).get('source_doc', '')})"
            for c in final_chunks
        ]

        trace_entry = trace_logger.log_step(
            agent_name="retriever",
            input_summary={"sub_queries": sub_queries, "filters": where_filter},
            output_summary={"retrieved_count": len(final_chunks), "chunks": chunk_summaries},
            tokens=0,  # Embedding tokens are logged inside openrouter_client
            latency_ms=latency_ms,
            metadata={"strategy": "hybrid_dense_bm25_rrf"}
        )

        print(f"[RETRIEVAL AGENT] Retrieved {len(final_chunks)} fused chunks across {len(sub_queries)} sub-queries:")
        for idx, c in enumerate(final_chunks, 1):
            meta = c.get("metadata", {})
            print(f"  [{idx}] {c['chunk_id']} | {meta.get('source_doc')} ({meta.get('section_header')}) | RRF: {c.get('rrf_score', 0):.4f}")

        return {
            "retrieved_chunks": final_chunks,
            "trace_logs": state.get("trace_logs", []) + [trace_entry]
        }

# Callable instance
retrieval_node = RetrievalNode()
