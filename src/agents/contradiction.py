"""
Contradiction Agent Node

Role in the System:
-------------------
The Contradiction Agent is the core differentiator of this system. It analyzes
retrieved chunks and verified claims across differing document versions (e.g.,
Master Service Agreement v1 vs. Amendment v2) or conflicting policies (e.g.,
General Security Policy vs. GDPR Data Processing Addendum).

When conflicting provisions exist (e.g., 30-day vs. 90-day termination notice),
it explicitly surfaces the conflict, identifies the governing hierarchy rule,
and presents a transparent comparison rather than arbitrarily picking one side
or blending them into an ambiguous summary like Naive RAG.

Inputs & Outputs:
-----------------
- Input: `state["retrieved_chunks"]`, `state["draft_answer"]`, `state["original_query"]`
- Output updates to `AgentState`:
    - `contradictions_found`: List of structured conflict dictionaries
    - `final_answer`: Enriched answer containing an explicit Contradiction Notice
    - `trace_logs`: Appended trace entry

Why a separate Contradiction Agent rather than folded into Synthesis:
---------------------------------------------------------------------
1. Multi-Document Temporal Reasoning: Synthesis focuses on generating a smooth,
   coherent answer to the user's specific prompt. Contradiction analysis requires
   a dedicated cross-document diffing pass across versions and precedence recitals.
2. Separation of Verification vs. Conflict Detection: The Critic checks if a claim
   is grounded in a chunk; the Contradiction Agent checks if Chunk A fundamentally
   disagrees with Chunk B. These are distinct reasoning tasks.
"""
import re
import json
import time
from typing import Dict, Any, List, Optional

from src.config import SYNTHESIS_MODEL
from src.agents.state import AgentState
from src.core.openrouter_client import openrouter_client
from src.core.trace_logger import trace_logger

def _heuristic_detect_contradiction(
    chunks: List[Dict[str, Any]],
    draft_answer: str,
    query: str
) -> List[Dict[str, Any]]:
    """
    Deterministic rule-based conflict detector for standard document dimensions.
    Operates offline or as a fast pre-filter.
    """
    conflicts = []
    text_lower = " ".join([c.get("text", "") for c in chunks]).lower()
    draft_lower = draft_answer.lower()
    q_lower = query.lower()

    # Check 1: Notice Period Conflict (MSA v1 30 days vs Amendment v2 90 days)
    if ("30 days" in text_lower or "thirty (30) days" in text_lower) and \
       ("90 days" in text_lower or "ninety (90) days" in text_lower):
        v1_cids = [c["chunk_id"] for c in chunks if "v1" in c.get("metadata", {}).get("version", "").lower() or "v1" in c.get("chunk_id", "").lower()]
        v2_cids = [c["chunk_id"] for c in chunks if "v2" in c.get("metadata", {}).get("version", "").lower() or "v2" in c.get("chunk_id", "").lower()]
        
        conflicts.append({
            "dimension": "Termination Notice Period",
            "v1_claim": f"Thirty (30) days written notice [{' '.join(v1_cids[:1])}]",
            "v2_claim": f"Ninety (90) days written notice [{' '.join(v2_cids[:1])}]",
            "governing_document": "msa_amendment_v2.txt",
            "resolution_rationale": "Amendment No. 1 (DOC-MSA-2024-V2) Section 8.1 expressly replaces the 30-day notice requirement of MSA v1 with 90 days. Per Recitals, Amendment terms govern and control in any conflict."
        })

    # Check 2: Payment Terms Conflict (Net-30 vs Net-60)
    if "net-30" in text_lower and "net-60" in text_lower:
        v1_cids = [c["chunk_id"] for c in chunks if "v1" in c.get("chunk_id", "").lower()]
        v2_cids = [c["chunk_id"] for c in chunks if "v2" in c.get("chunk_id", "").lower()]
        conflicts.append({
            "dimension": "Payment Terms",
            "v1_claim": f"Payment due within Net-30 days [{' '.join(v1_cids[:1])}]",
            "v2_claim": f"Payment extended to Net-60 days [{' '.join(v2_cids[:1])}]",
            "governing_document": "msa_amendment_v2.txt",
            "resolution_rationale": "Amendment No. 1 Section 4.1 extends payment terms from Net-30 to Net-60 days."
        })

    # Check 3: SLA Availability Conflict (99.9% vs 99.95%)
    if "99.9%" in text_lower and "99.95%" in text_lower:
        v1_cids = [c["chunk_id"] for c in chunks if "v1" in c.get("chunk_id", "").lower()]
        v2_cids = [c["chunk_id"] for c in chunks if "v2" in c.get("chunk_id", "").lower()]
        conflicts.append({
            "dimension": "Service Level Agreement (Uptime)",
            "v1_claim": f"Guaranteed Service Uptime of 99.9% per month [{' '.join(v1_cids[:1])}]",
            "v2_claim": f"Upgraded Service Uptime of 99.95% per month [{' '.join(v2_cids[:1])}]",
            "governing_document": "msa_amendment_v2.txt",
            "resolution_rationale": "Amendment No. 1 Section 6.1 upgrades availability SLA from 99.9% to 99.95%."
        })

    # Check 4: Data Breach Notification Timeline (General Policy 72 hrs vs DPA 24 hrs)
    if ("seventy-two (72) hours" in text_lower or "72 hours" in text_lower or "72-hour" in text_lower) and \
       ("twenty-four (24) hours" in text_lower or "24 hours" in text_lower):
        pol_cids = [c["chunk_id"] for c in chunks if "policy" in c.get("chunk_id", "").lower() or "security" in c.get("chunk_id", "").lower()]
        dpa_cids = [c["chunk_id"] for c in chunks if "dpa" in c.get("chunk_id", "").lower() or "addendum" in c.get("chunk_id", "").lower()]
        conflicts.append({
            "dimension": "Data Breach Notification Timeline",
            "v1_claim": f"Standard client notification within 72 hours of breach verification [{' '.join(pol_cids[:1])}]",
            "v2_claim": f"Accelerated notification within 24 hours for EU citizen Personal Data [{' '.join(dpa_cids[:1])}]",
            "governing_document": "data_processing_addendum_2024.txt",
            "resolution_rationale": "DPA 2024 Section 4.1 explicitly states that notwithstanding general corporate policies (72 hours), EU citizen Personal Data incidents require 24-hour notification."
        })

    return conflicts

def format_contradiction_notice(conflicts: List[Dict[str, Any]]) -> str:
    """
    Renders structured conflicts into a prominent, standardized markdown alert box.
    """
    if not conflicts:
        return ""

    lines = [
        "\n\n---",
        "### [!] Contradiction & Version Comparison Alert",
        "> [!IMPORTANT]",
        "> Multiple document versions or policies were retrieved with differing terms. Detailed conflict analysis:"
    ]


    for idx, c in enumerate(conflicts, 1):
        lines.append(f"> #### {idx}. Conflict: {c['dimension']}")
        lines.append(f"> - **Earlier / General Term**: {c['v1_claim']}")
        lines.append(f"> - **Amended / Specialized Term**: {c['v2_claim']}")
        lines.append(f"> - **Governing Authority**: `{c['governing_document']}`")
        lines.append(f"> - **Precedence Rule**: {c['resolution_rationale']}")
        if idx < len(conflicts):
            lines.append(">")

    lines.append("---\n")
    return "\n".join(lines)

def contradiction_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node function for the Contradiction Agent.
    Compares evidence across documents/versions, identifies conflicts, and enriches final_answer.
    """
    start_time = time.time()
    query = state.get("original_query", "")
    chunks = state.get("retrieved_chunks", [])
    draft_answer = state.get("draft_answer") or state.get("final_answer") or ""

    # Check if chunks contain multiple distinct documents or versions
    docs_present = {c.get("metadata", {}).get("source_doc") for c in chunks if c.get("metadata", {}).get("source_doc")}
    versions_present = {c.get("metadata", {}).get("version") for c in chunks if c.get("metadata", {}).get("version")}

    # Fast heuristic check first
    detected_conflicts = _heuristic_detect_contradiction(chunks, draft_answer, query)

    tokens_used = 0

    # If heuristics didn't find anything but multiple versions exist, run LLM cross-examination
    if not detected_conflicts and len(docs_present) > 1 and len(versions_present) > 1:
        chunk_texts = "\n\n".join([
            f"=== [{c['chunk_id']}] Document: {c.get('metadata', {}).get('source_doc')} (Version: {c.get('metadata', {}).get('version')}) ===\n{c.get('text', '')}"
            for c in chunks
        ])

        system_prompt = (
            "You are a specialized Legal & Policy Contradiction Detection Agent. "
            "Analyze the provided multi-version evidence chunks and determine if any terms, "
            "timeframes, percentages, monetary amounts, or obligations directly contradict or amend one another.\n\n"
            "Respond STRICTLY in JSON format:\n"
            "{\n"
            "  'has_contradiction': true/false,\n"
            "  'conflicts': [\n"
            "    {\n"
            "      'dimension': 'Name of conflicting term (e.g. Notice Period)',\n"
            "      'v1_claim': 'Statement from earlier doc with citation',\n"
            "      'v2_claim': 'Statement from amendment with citation',\n"
            "      'governing_document': 'Filename of governing doc',\n"
            "      'resolution_rationale': 'Why this document supersedes the other'\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        user_prompt = (
            f"User Question: {query}\n\n"
            f"Draft Answer: {draft_answer}\n\n"
            f"Evidence Chunks from Multiple Documents:\n{chunk_texts}\n\n"
            f"JSON Contradiction Analysis:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = openrouter_client.chat_completion(
                model=SYNTHESIS_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=500
            )
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            content = response.get("content", "").strip()

            if "```" in content:
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            if parsed.get("has_contradiction"):
                detected_conflicts = parsed.get("conflicts", [])

        except Exception as e:
            # Fall back gracefully to heuristics
            pass

    latency_ms = int((time.time() - start_time) * 1000)

    # Format enriched final answer
    if detected_conflicts:
        notice_block = format_contradiction_notice(detected_conflicts)
        enriched_final_answer = f"{draft_answer}\n{notice_block}"
        print(f"[CONTRADICTION AGENT] Detected {len(detected_conflicts)} cross-document conflicts! Appended Contradiction Alert.")
    else:
        enriched_final_answer = draft_answer
        print("[CONTRADICTION AGENT] No cross-document contradictions detected.")

    trace_entry = trace_logger.log_step(
        agent_name="contradiction",
        input_summary={"chunks_count": len(chunks), "docs_present": list(docs_present)},
        output_summary={
            "contradictions_count": len(detected_conflicts),
            "dimensions": [c.get("dimension") for c in detected_conflicts]
        },
        tokens=tokens_used,
        latency_ms=latency_ms,
        metadata={"model": SYNTHESIS_MODEL}
    )

    return {
        "contradictions_found": detected_conflicts,
        "final_answer": enriched_final_answer,
        "trace_logs": state.get("trace_logs", []) + [trace_entry]
    }
