"""
LangGraph Multi-Agent State Machine

Role in the System:
-------------------
Defines and compiles the core multi-agent state graph connecting all 5 agents:
Orchestrator -> Retrieval -> Synthesis -> Verification Critic -> Contradiction Analyzer.

Phase 5 Complete Graph Topology:
--------------------------------
                    [START]
                       │
                       ▼
               ┌───────────────┐
               │ orchestrator  │
               └───────┬───────┘
                       │
             needs_retrieval?
            ┌──────────┴──────────┐
      [True]│                     │[False]
            ▼                     │
     ┌─────────────┐              │
 ┌──►│  retriever  │              │
 │   └──────┬──────┘              │
 │          │                     │
 │          └──────────┬──────────┘
 │                     │
 │                     ▼
 │             ┌───────────────┐
 │             │   synthesis   │
 │             └───────┬───────┘
 │                     │
 │                     ▼
 │             ┌───────────────┐
 │             │    critic     │
 │             └───────┬───────┘
 │                     │
 │           verification_passed?
 │           [False]   │   [True]
 └─────────────────────┤
                       ▼
             ┌───────────────────┐
             │   contradiction   │
             └─────────┬─────────┘
                       │
                       ▼
                     [END]
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.orchestrator import orchestrator_node
from src.agents.retrieval import retrieval_node
from src.agents.synthesis import synthesis_node
from src.agents.critic import critic_node
from src.agents.contradiction import contradiction_node

def route_orchestrator(state: AgentState) -> str:
    """
    Conditional routing function:
    Directs flow to 'retriever' if documents are required,
    or bypasses retrieval straight to 'synthesis' for conversational queries.
    """
    if state.get("needs_retrieval", True):
        return "retriever"
    return "synthesis"

def route_critic(state: AgentState) -> str:
    """
    Conditional routing function for the Verification Critic:
    - If verification passed (or attempts capped): proceed to 'contradiction' agent.
    - If verification failed: route BACK to 'retriever' for self-correction.
    """
    if state.get("verification_passed", True):
        return "contradiction"
    return "retriever"

def build_graph() -> StateGraph:
    """Constructs the full 5-node LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # 1. Register All 5 Agent Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("contradiction", contradiction_node)

    # 2. Add Edges & Conditional Branches
    workflow.add_edge(START, "orchestrator")

    # Orchestrator Branching
    workflow.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "retriever": "retriever",
            "synthesis": "synthesis"
        }
    )

    workflow.add_edge("retriever", "synthesis")
    workflow.add_edge("synthesis", "critic")

    # Critic Self-Correction Branching
    workflow.add_conditional_edges(
        "critic",
        route_critic,
        {
            "contradiction": "contradiction",
            "retriever": "retriever"
        }
    )

    # Contradiction to End
    workflow.add_edge("contradiction", END)

    return workflow

# Compile executable state graph application
agent_app = build_graph().compile()

def run_agent_system(query: str) -> AgentState:
    """
    Convenience execution helper that initializes state and runs the graph.
    """
    initial_state: AgentState = {
        "original_query": query,
        "sub_queries": [],
        "needs_retrieval": True,
        "retrieval_filters": None,
        "retrieved_chunks": [],
        "draft_answer": None,
        "citations": [],
        "verification_passed": False,
        "critic_feedback": None,
        "verification_attempts": 0,
        "contradictions_found": [],
        "final_answer": None,
        "trace_logs": []
    }

    final_state = agent_app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    test_query = "What is the termination notice period under the contract and its amendment?"
    print(f"Running full 5-node agent graph for: '{test_query}'\n")
    output = run_agent_system(test_query)
    print("\nFINAL ANSWER:\n", output.get("final_answer"))
    print("\nCONTRADICTIONS FOUND:\n", output.get("contradictions_found"))
