"""
Evaluation Metrics Module (Pure Python, No Heavy Frameworks)

Role in the System:
-------------------
Implements transparent, mathematical evaluation metrics for measuring:
1. Retrieval Quality (Precision & Recall against labeled ground-truth chunks)
2. Faithfulness / Grounding (Deterministic verification of citations)
3. Contradiction Detection Accuracy (Correct detection & precedence resolution)

Why Hand-Rolled Metrics instead of RAGAS:
-----------------------------------------
1. No Dependency Bloat: RAGAS pulls in dozens of heavy transitive packages.
2. Deterministic & Explainable: Instead of asking an LLM to grade another LLM
   with a subjective prompt, our metrics evaluate exact chunk IDs, mathematical
   set intersections, and strict citation fidelity.
"""
from typing import List, Dict, Any, Tuple, Optional

def compute_retrieval_metrics(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str]
) -> Tuple[float, float]:
    """
    Computes Precision and Recall of retrieval against ground-truth chunk IDs.
    Returns: (precision, recall)
    """
    if not ground_truth_chunk_ids:
        return 1.0, 1.0
    if not retrieved_chunk_ids:
        return 0.0, 0.0

    retrieved_set = set(retrieved_chunk_ids)
    gt_set = set(ground_truth_chunk_ids)

    intersection = retrieved_set.intersection(gt_set)
    precision = len(intersection) / len(retrieved_set)
    recall = len(intersection) / len(gt_set)

    return round(precision, 4), round(recall, 4)

def compute_faithfulness(
    cited_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    answer_text: str
) -> float:
    """
    Computes Faithfulness / Citation Grounding score (0.0 to 1.0):
    1. If citations are present, checks what fraction exist in retrieved_chunk_ids (no hallucinated chunk IDs).
    2. Penalizes answers that produce no citations when evidence was retrieved.
    """
    if not answer_text.strip():
        return 0.0

    if not retrieved_chunk_ids:
        # Conversational / non-retrieval query
        return 1.0

    if not cited_chunk_ids:
        # Failed to cite any sources despite retrieved context
        return 0.2

    retrieved_set = set(retrieved_chunk_ids)
    valid_citations = [cid for cid in cited_chunk_ids if cid in retrieved_set]
    
    # Ratio of valid cited chunks out of total cited chunks
    citation_validity_ratio = len(valid_citations) / len(cited_chunk_ids)
    return round(citation_validity_ratio, 4)

def compute_contradiction_accuracy(
    detected_conflicts: List[Dict[str, Any]],
    expected_has_contradiction: bool,
    expected_governing_doc: Optional[str]
) -> float:
    """
    Evaluates whether the system accurately identified conflicts and resolved precedence:
    - If no contradiction was expected: 1.0 if none detected, 0.0 if false positive.
    - If contradiction was expected:
      - 0.0 if not detected (false negative)
      - 0.5 if detected, but failed to identify governing authority
      - 1.0 if detected AND resolved the correct governing document
    """
    has_detected = len(detected_conflicts) > 0

    if not expected_has_contradiction:
        return 1.0 if not has_detected else 0.0

    if not has_detected:
        return 0.0

    # Contradiction detected; verify governing document
    if expected_governing_doc:
        for c in detected_conflicts:
            gov = c.get("governing_document", "")
            if expected_governing_doc.lower() in gov.lower():
                return 1.0
        return 0.5  # Flagged conflict, but governing doc mismatched

    return 1.0
