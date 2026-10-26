"""
LangGraph Shared State Schema

Role in the System:
-------------------
In LangGraph, all agents (nodes) read from and write to a single, strongly-typed
shared state. This file defines `AgentState` as a TypedDict.

Why a centralized TypedDict rather than ad-hoc state objects:
------------------------------------------------------------
1. Type safety: Graph nodes have explicit contracts about what keys exist.
2. Immutability by convention: Nodes return dictionary updates rather than
   mutating state in-place, enabling predictable state transitions and
   easy time-travel/debugging.
3. Forward compatibility: Future nodes (Phase 4 Critic, Phase 5 Contradiction)
   are declared in the schema today so downstream edges and state inspection
   remain consistent across phases.

State Lifecycle:
----------------
[original_query]
      |
[orchestrator_node] -> sub_queries, needs_retrieval, retrieval_filters
      |
[retrieval_node]    -> retrieved_chunks
      |
[synthesis_node]    -> draft_answer, citations, final_answer
"""
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    # User Input
    original_query: str

    # Orchestrator Decisions
    sub_queries: List[str]
    needs_retrieval: bool
    retrieval_filters: Optional[Dict[str, Any]]

    # Retrieval Output
    retrieved_chunks: List[Dict[str, Any]]

    # Synthesis Output
    draft_answer: Optional[str]
    citations: List[str]  # List of chunk_ids cited in the answer

    # Verification / Critic Output (Phase 4 forward-compatibility)
    verification_passed: bool
    critic_feedback: Optional[str]
    verification_attempts: int

    # Contradiction Detection Output (Phase 5 forward-compatibility)
    contradictions_found: List[Dict[str, Any]]

    # Final Output Delivered to User
    final_answer: Optional[str]

    # Observability & Trace Logging
    trace_logs: List[Dict[str, Any]]
