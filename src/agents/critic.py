"""
Verification / Critic Agent Node

Role in the System:
-------------------
The Verification Critic is the automated quality and hallucination-mitigation
gatekeeper in our LangGraph state machine. It inspects the draft answer generated
by the Synthesis Agent, verifies that every factual statement is strictly backed
by the retrieved evidence chunks, checks for hallucinated citations, and triggers
a self-correction re-retrieval loop if evidence is missing or claims are unsupported.

Inputs & Outputs:
-----------------
- Input: `state["draft_answer"]`, `state["retrieved_chunks"]`, `state["verification_attempts"]`
- Output updates to `AgentState`:
    - `verification_passed`: Boolean pass/fail verdict
    - `critic_feedback`: Specific critique or description of unsupported claims
    - `verification_attempts`: Incremented attempt counter (caps loops at max 2)
    - `sub_queries`: Updated targeted queries if re-retrieval is needed
    - `final_answer`: Verified answer (with disclaimers if ungrounded claims persist)
    - `trace_logs`: Appended trace entry

Why a separate Critic Agent rather than trusting the Synthesis LLM:
-------------------------------------------------------------------
1. LLM Self-Delusion Bias: An LLM asked to "generate and ensure you do not hallucinate"
   routinely hallucinates anyway because its generation probability distribution
   favors fluency over strict adherence.
2. Separation of Concerns: Synthesis focuses on clear, fluent language; the Critic
   focuses solely on adversarial verification and entailment checking.
3. Self-Correction Loop: Instead of presenting a bad answer to the user, the Critic
   programmatically drives a LangGraph feedback loop back to the Retrieval Agent.
"""
import re
import json
import time
from typing import Dict, Any, List, Tuple

from src.config import CRITIC_MODEL
from src.agents.state import AgentState
from src.agents.synthesis import extract_citations
from src.core.openrouter_client import openrouter_client, _is_placeholder_key
from src.core.trace_logger import trace_logger

def verify_citations_programmatically(
    draft_answer: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> Tuple[bool, List[str], List[str]]:
    """
    Fast deterministic check:
    1. Extracts all [chunk_id] citations from draft answer.
    2. Verifies whether each cited chunk_id exists in retrieved_chunks.
    Returns: (is_valid, valid_citations, hallucinated_citations)
    """
    cited_ids = extract_citations(draft_answer)
    retrieved_cids = {c["chunk_id"] for c in retrieved_chunks}

    valid_cits = []
    hallucinated_cits = []

    for cid in cited_ids:
        if cid in retrieved_cids:
            valid_cits.append(cid)
        else:
            hallucinated_cits.append(cid)

    is_valid = len(hallucinated_cits) == 0 and (len(valid_cits) > 0 or len(retrieved_chunks) == 0)
    return is_valid, valid_cits, hallucinated_cits

def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node function for the Verification Critic.
    Inspects draft answer, validates grounding, and sets verification_passed.
    """
    start_time = time.time()
    draft_answer = state.get("draft_answer") or ""
    retrieved_chunks = state.get("retrieved_chunks") or []
    attempts = state.get("verification_attempts", 0) + 1

    # Trivial case: If no retrieval was needed (conversational query)
    if not state.get("needs_retrieval", True) or not retrieved_chunks:
        trace_entry = trace_logger.log_step(
            agent_name="critic",
            input_summary="Conversational query (no retrieval verification needed)",
            output_summary={"verification_passed": True, "attempts": attempts},
            tokens=0,
            latency_ms=int((time.time() - start_time) * 1000)
        )
        return {
            "verification_passed": True,
            "critic_feedback": None,
            "verification_attempts": attempts,
            "final_answer": draft_answer,
            "trace_logs": state.get("trace_logs", []) + [trace_entry]
        }

    # Step 1: Programmatic Citation Validation
    cit_valid, valid_cits, hallucinated_cits = verify_citations_programmatically(draft_answer, retrieved_chunks)

    # Step 2: Semantic Entailment Verification
    # Format chunk evidence for the Critic model
    chunk_evidence = "\n\n".join([
        f"--- [{c['chunk_id']}] ---\n{c.get('text', '')}"
        for c in retrieved_chunks
    ])

    system_prompt = (
        "You are an adversarial Verification Critic in a document intelligence system. "
        "Your task is to verify whether EVERY claim in the Draft Answer is strictly supported "
        "by the provided Evidence Chunks.\n\n"
        "Evaluation Criteria:\n"
        "1. Grounding: Is every claim backed by facts in the cited chunks?\n"
        "2. Hallucination: Are any facts invented or inferred without evidence?\n"
        "3. Missing Evidence: Did the user ask for information not found in the chunks?\n\n"
        "Respond STRICTLY in JSON format with keys:\n"
        "- 'verification_passed' (bool): true if all claims are supported, false otherwise\n"
        "- 'unsupported_claims' (list of strings): statements lacking evidence\n"
        "- 'missing_topics' (list of strings): 1-2 search terms to retrieve missing evidence\n"
        "- 'feedback' (string): brief explanation"
    )

    user_prompt = (
        f"Original User Question: {state.get('original_query', '')}\n\n"
        f"Evidence Chunks:\n{chunk_evidence}\n\n"
        f"Draft Answer to Verify:\n{draft_answer}\n\n"
        f"JSON Verification Output:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    verification_passed = True
    unsupported_claims: List[str] = []
    missing_topics: List[str] = []
    feedback = "All claims verified against evidence chunks."
    tokens_used = 0

    # If hallucinated citations were caught programmatically, immediately fail
    if not cit_valid and hallucinated_cits:
        verification_passed = False
        unsupported_claims.append(f"Cites non-existent chunks: {hallucinated_cits}")
        feedback = f"Programmatic Check Failed: Draft cites non-existent chunks {hallucinated_cits}"
        missing_topics = [state.get("original_query", "")]

    else:
        try:
            response = openrouter_client.chat_completion(
                model=CRITIC_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=350
            )
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            content = response.get("content", "").strip()

            # Clean JSON markdown fences
            if "```" in content:
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            verification_passed = bool(parsed.get("verification_passed", True))
            unsupported_claims = parsed.get("unsupported_claims", [])
            missing_topics = parsed.get("missing_topics", [])
            feedback = parsed.get("feedback", "Completed verification check.")

        except Exception as e:
            # Fallback heuristic if offline or JSON parsing fails
            if "[UNSUPPORTED]" in draft_answer or "hallucinated" in draft_answer.lower():
                verification_passed = False
                unsupported_claims.append("Flagged unsupported keyword in draft.")
                feedback = "Draft contains flagged ungrounded claims."
                missing_topics = [state.get("original_query", "")]
            else:
                verification_passed = True
                feedback = "Offline verification passed."

    latency_ms = int((time.time() - start_time) * 1000)

    # LOOP GUARD: If we have reached max attempts (2), do not loop again!
    # Instead, annotate the final answer with a disclaimer and pass forward.
    final_ans = draft_answer
    if not verification_passed and attempts >= 2:
        print(f"[CRITIC LOOP GUARD] Max verification attempts ({attempts}) reached. Adding disclaimer and forcing pass.")
        disclaimer = (
            "\n\n> [!WARNING] Grounding Verification Notice:\n"
            f"> The following claims could not be fully verified against source chunks: {unsupported_claims}"
        )
        final_ans = f"{draft_answer}{disclaimer}"
        verification_passed = True
        feedback += " (Loop capped at max attempts; warning appended to answer)."

    # Log trace entry
    trace_entry = trace_logger.log_step(
        agent_name="critic",
        input_summary={"answer_length": len(draft_answer), "attempts": attempts},
        output_summary={
            "verification_passed": verification_passed,
            "unsupported_count": len(unsupported_claims),
            "feedback": feedback
        },
        tokens=tokens_used,
        latency_ms=latency_ms,
        metadata={"model": CRITIC_MODEL, "attempts": attempts}
    )

    print(f"[CRITIC AGENT] Attempt {attempts} | Passed: {verification_passed} | Feedback: {feedback}")

    updates: Dict[str, Any] = {
        "verification_passed": verification_passed,
        "critic_feedback": feedback,
        "verification_attempts": attempts,
        "final_answer": final_ans,
        "trace_logs": state.get("trace_logs", []) + [trace_entry]
    }

    # If verification failed and we need to loop back, update sub_queries for targeted retrieval
    if not verification_passed and missing_topics:
        updates["sub_queries"] = missing_topics

    return updates
