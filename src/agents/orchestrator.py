"""
Orchestrator Agent Node

Role in the System:
-------------------
The Orchestrator is the entry decision-maker in our LangGraph state machine.
It receives the raw user query, analyzes whether document retrieval is required,
and decomposes complex or multi-clause queries into 1-3 targeted sub-queries.

Inputs & Outputs:
-----------------
- Input: `state["original_query"]`
- Output updates to `AgentState`:
    - `sub_queries`: List of atomic search queries
    - `needs_retrieval`: Boolean routing flag
    - `retrieval_filters`: Optional metadata filters
    - `trace_logs`: Appended trace entry

Why a separate Orchestrator rather than passing the query straight to retrieval:
--------------------------------------------------------------------------------
1. Comparative queries (e.g., "Compare the termination notice between MSA v1 and Amendment v2")
   fail in single-shot RAG because a single embedding searches for a conceptual midpoint.
   Decomposing it into "termination notice MSA v1" and "termination notice amendment v2"
   guarantees both documents are retrieved.
2. Query routing: Conversational greetings ("Hello", "Thank you") or meta-questions
   can bypass retrieval entirely, saving token budget and latency.
"""
import json
import re
import time
from typing import Dict, Any, List

from src.config import ORCHESTRATOR_MODEL
from src.agents.state import AgentState
from src.core.openrouter_client import openrouter_client, _is_placeholder_key
from src.core.trace_logger import trace_logger

def _heuristic_decomposition(query: str) -> List[str]:
    """
    Rule-based decomposition fallback when offline or for fast processing.
    Splits on 'and', 'vs', 'compared to', 'difference between'.
    """
    q_lower = query.lower()
    
    # Check for comparative queries
    if "compare" in q_lower or "difference between" in q_lower or " vs " in q_lower:
        parts = re.split(r"\b(?:and|vs|versus|compared to)\b", query, flags=re.IGNORECASE)
        cleaned = [p.strip() for p in parts if len(p.strip()) > 3]
        if len(cleaned) > 1:
            return cleaned

    # Check for multi-aspect queries (e.g. "notice period and payment terms")
    if " and " in q_lower:
        parts = query.split(" and ")
        if len(parts) == 2 and len(parts[0]) > 5 and len(parts[1]) > 5:
            return [parts[0].strip(), parts[1].strip()]

    return [query]

def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node function for the Orchestrator.
    Reads state and returns dictionary of state updates.
    """
    start_time = time.time()
    query = state.get("original_query", "").strip()

    # Trivial / conversational check (skip retrieval if not asking a document question)
    greetings = ["hello", "hi", "hey", "thanks", "thank you", "who are you"]
    if query.lower() in greetings:
        trace_entry = trace_logger.log_step(
            agent_name="orchestrator",
            input_summary=query,
            output_summary={"needs_retrieval": False, "sub_queries": []},
            tokens=0,
            latency_ms=int((time.time() - start_time) * 1000)
        )
        return {
            "needs_retrieval": False,
            "sub_queries": [],
            "retrieval_filters": None,
            "trace_logs": state.get("trace_logs", []) + [trace_entry]
        }

    # Prompt Orchestrator model to decompose query
    system_prompt = (
        "You are an expert query planning orchestrator for a document intelligence system. "
        "Analyze the user's question and determine:\n"
        "1. Does it require document retrieval? (needs_retrieval: true/false)\n"
        "2. Break down complex or comparative queries into 1 to 3 atomic sub-queries that can be retrieved independently.\n"
        "Respond STRICTLY in JSON format with keys: 'needs_retrieval' (bool) and 'sub_queries' (list of strings)."
    )

    user_prompt = f"User Question: {query}\n\nJSON Output:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    tokens_used = 0
    sub_queries = [query]
    needs_retrieval = True

    try:
        response = openrouter_client.chat_completion(
            model=ORCHESTRATOR_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=250
        )
        tokens_used = response.get("usage", {}).get("total_tokens", 0)
        content = response.get("content", "").strip()

        # Parse JSON output from model
        # Strip potential markdown formatting (```json ... ```)
        if "```" in content:
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        parsed = json.loads(content)
        needs_retrieval = parsed.get("needs_retrieval", True)
        parsed_sub = parsed.get("sub_queries", [])
        if parsed_sub and isinstance(parsed_sub, list):
            sub_queries = [str(s).strip() for s in parsed_sub if str(s).strip()]
        else:
            sub_queries = _heuristic_decomposition(query)

    except Exception as e:
        # Graceful fallback to heuristic decomposition
        sub_queries = _heuristic_decomposition(query)
        needs_retrieval = True

    latency_ms = int((time.time() - start_time) * 1000)

    trace_entry = trace_logger.log_step(
        agent_name="orchestrator",
        input_summary=query,
        output_summary={"needs_retrieval": needs_retrieval, "sub_queries": sub_queries},
        tokens=tokens_used,
        latency_ms=latency_ms,
        metadata={"model": ORCHESTRATOR_MODEL}
    )

    print(f"[ORCHESTRATOR] Needs Retrieval: {needs_retrieval} | Sub-queries ({len(sub_queries)}): {sub_queries}")

    return {
        "needs_retrieval": needs_retrieval,
        "sub_queries": sub_queries,
        "retrieval_filters": None,
        "trace_logs": state.get("trace_logs", []) + [trace_entry]
    }
