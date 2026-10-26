"""
Phase 6 Verification Script
Validates metric math, evaluation data integrity, and executes the complete
comparative benchmark between Naive Single-Shot RAG and the Multi-Agent System.
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.metrics import (
    compute_retrieval_metrics,
    compute_faithfulness,
    compute_contradiction_accuracy
)
from src.eval.evaluator import BenchmarkEvaluator, EVAL_SET_PATH
from scripts.run_benchmark import update_readme_table

def verify_phase6():
    print("=" * 75)
    print("Phase 6 Verification Check: Eval Harness & Benchmark Evaluation")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Test Retrieval Precision & Recall Math
    # -------------------------------------------------------------
    print("\n[CHECK 1] Testing Pure-Python Retrieval Metric Calculations...")
    gt = ["chunk_A", "chunk_B"]
    ret_full = ["chunk_A", "chunk_B", "chunk_C", "chunk_D"]
    ret_partial = ["chunk_A", "chunk_C"]

    p1, r1 = compute_retrieval_metrics(ret_full, gt)
    assert p1 == 0.5, f"FAILED: Expected precision 0.5, got {p1}"
    assert r1 == 1.0, f"FAILED: Expected recall 1.0, got {r1}"

    p2, r2 = compute_retrieval_metrics(ret_partial, gt)
    assert p2 == 0.5, f"FAILED: Expected precision 0.5, got {p2}"
    assert r2 == 0.5, f"FAILED: Expected recall 0.5, got {r2}"
    print(f"  [PASS] Precision & Recall calculations verified.")

    # -------------------------------------------------------------
    # 2. Test Faithfulness / Grounding Metric
    # -------------------------------------------------------------
    print("\n[CHECK 2] Testing Faithfulness & Citation Grounding Calculations...")
    retrieved = ["doc_chunk_1", "doc_chunk_2"]
    
    # 100% grounded (all cited chunks were retrieved)
    f_high = compute_faithfulness(["doc_chunk_1"], retrieved, "Answer text [doc_chunk_1]")
    assert f_high == 1.0, f"FAILED: Expected faithfulness 1.0, got {f_high}"

    # Hallucinated citation
    f_low = compute_faithfulness(["fake_chunk_99"], retrieved, "Answer text [fake_chunk_99]")
    assert f_low == 0.0, f"FAILED: Expected faithfulness 0.0, got {f_low}"

    # No citations despite retrieved evidence
    f_none = compute_faithfulness([], retrieved, "Answer text without citations")
    assert f_none == 0.2, f"FAILED: Expected penalty score 0.2, got {f_none}"
    print(f"  [PASS] Faithfulness metric correctly verifies grounding and penalizes hallucinations.")

    # -------------------------------------------------------------
    # 3. Test Contradiction Detection Accuracy Metric
    # -------------------------------------------------------------
    print("\n[CHECK 3] Testing Contradiction Accuracy Calculations...")
    conflicts = [{"dimension": "Notice", "governing_document": "msa_amendment_v2.txt"}]
    
    # Correct detection & governing resolution
    acc_perfect = compute_contradiction_accuracy(conflicts, True, "msa_amendment_v2.txt")
    assert acc_perfect == 1.0, f"FAILED: Expected accuracy 1.0, got {acc_perfect}"

    # Correct negative (no conflict expected, none detected)
    acc_neg = compute_contradiction_accuracy([], False, None)
    assert acc_neg == 1.0, f"FAILED: Expected accuracy 1.0, got {acc_neg}"

    # False negative (conflict expected, none detected - Naive RAG behavior)
    acc_fn = compute_contradiction_accuracy([], True, "msa_amendment_v2.txt")
    assert acc_fn == 0.0, f"FAILED: Expected accuracy 0.0, got {acc_fn}"
    print(f"  [PASS] Contradiction accuracy metric verified.")

    # -------------------------------------------------------------
    # 4. Run Live Benchmark Evaluation over Golden Dataset
    # -------------------------------------------------------------
    print("\n[CHECK 4] Running Comparative Benchmark over Golden Eval Set...")
    evaluator = BenchmarkEvaluator()
    summary = evaluator.evaluate()

    n = summary["naive_rag"]
    m = summary["multi_agent"]

    print("\n  Summary Metrics:")
    print(f"    Retrieval Recall:       Naive {n['avg_retrieval_recall']*100:.1f}% vs Multi-Agent {m['avg_retrieval_recall']*100:.1f}%")
    print(f"    Faithfulness:           Naive {n['avg_faithfulness']*100:.1f}% vs Multi-Agent {m['avg_faithfulness']*100:.1f}%")
    print(f"    Contradiction Accuracy: Naive {n['contradiction_accuracy']*100:.1f}% vs Multi-Agent {m['contradiction_accuracy']*100:.1f}%")

    # Assertions demonstrating Multi-Agent superiority
    assert m["avg_retrieval_recall"] >= n["avg_retrieval_recall"], "FAILED: Multi-Agent recall lower than Naive RAG!"
    assert m["contradiction_accuracy"] > n["contradiction_accuracy"], "FAILED: Multi-Agent did not outperform Naive RAG on contradiction detection!"
    assert m["avg_faithfulness"] >= 0.8, "FAILED: Multi-Agent faithfulness below 80%!"

    # -------------------------------------------------------------
    # 5. Format and Update README.md Benchmark Table
    # -------------------------------------------------------------
    print("\n[CHECK 5] Updating README.md with Live Benchmark Results Table...")
    table = evaluator.format_comparison_table(summary)
    update_readme_table(table)
    print("  [PASS] README.md updated with live evaluation table.")

    print("\n" + "=" * 75)
    print("Phase 6 Verification: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    verify_phase6()
