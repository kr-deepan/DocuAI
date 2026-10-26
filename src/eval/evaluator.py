"""
Benchmark Evaluator Module

Role in the System:
-------------------
Coordinates side-by-side evaluation of:
1. Baseline: Naive Single-Shot RAG Pipeline
2. Target: Complete 5-Node LangGraph Multi-Agent System

Evaluates both pipelines over the 6-question golden eval set and computes
average Precision, Recall, Faithfulness, Contradiction Accuracy, Latency, and Token Cost.
"""
import os
import json
import time
from typing import Dict, Any, List

from src.config import BASE_DIR
from src.baselines.naive_rag import NaiveRAGPipeline
from src.graph import run_agent_system
from src.agents.synthesis import extract_citations
from src.eval.metrics import (
    compute_retrieval_metrics,
    compute_faithfulness,
    compute_contradiction_accuracy
)

EVAL_SET_PATH = os.path.join(BASE_DIR, "data", "eval_set.json")

class BenchmarkEvaluator:
    def __init__(self, eval_set_path: str = EVAL_SET_PATH):
        self.eval_set_path = eval_set_path
        self.eval_data = self._load_eval_set()
        self.naive_pipeline = NaiveRAGPipeline(top_k=4)

    def _load_eval_set(self) -> List[Dict[str, Any]]:
        with open(self.eval_set_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self) -> Dict[str, Any]:
        """
        Executes benchmark comparison across all questions in the evaluation set.
        """
        print("=" * 75)
        print(f"Starting Benchmark Evaluation over {len(self.eval_data)} Golden QA Pairs")
        print("=" * 75)

        naive_scores = {
            "precision": [],
            "recall": [],
            "faithfulness": [],
            "contradiction_accuracy": [],
            "latency_ms": [],
            "tokens": []
        }

        multi_agent_scores = {
            "precision": [],
            "recall": [],
            "faithfulness": [],
            "contradiction_accuracy": [],
            "latency_ms": [],
            "tokens": []
        }

        detailed_results = []

        for idx, item in enumerate(self.eval_data, 1):
            qid = item["question_id"]
            query = item["query"]
            gt_chunks = item["ground_truth_chunks"]
            has_conflict = item["has_contradiction"]
            gov_doc = item["governing_document"]

            print(f"\n[{idx}/{len(self.eval_data)}] Evaluating: '{query}'")

            # ---------------------------------------------------------
            # 1. Evaluate Naive RAG Baseline
            # ---------------------------------------------------------
            naive_res = self.naive_pipeline.query(query)
            naive_retrieved_cids = [c["chunk_id"] for c in naive_res.get("retrieved_chunks", [])]
            naive_answer = naive_res.get("answer", "")
            naive_cited_cids = extract_citations(naive_answer)

            # Contradictions in Naive RAG: Naive RAG has no contradiction detector (conflicts = [])
            naive_prec, naive_rec = compute_retrieval_metrics(naive_retrieved_cids, gt_chunks)
            naive_faith = compute_faithfulness(naive_cited_cids, naive_retrieved_cids, naive_answer)
            naive_contra = compute_contradiction_accuracy([], has_conflict, gov_doc)

            naive_tokens = naive_res.get("usage", {}).get("total_tokens", 0)
            naive_latency = naive_res.get("latency_ms", 0)

            naive_scores["precision"].append(naive_prec)
            naive_scores["recall"].append(naive_rec)
            naive_scores["faithfulness"].append(naive_faith)
            naive_scores["contradiction_accuracy"].append(naive_contra)
            naive_scores["latency_ms"].append(naive_latency)
            naive_scores["tokens"].append(naive_tokens)

            # ---------------------------------------------------------
            # 2. Evaluate Multi-Agent LangGraph Graph
            # ---------------------------------------------------------
            start_time = time.time()
            ma_state = run_agent_system(query)
            ma_latency = int((time.time() - start_time) * 1000)

            ma_retrieved_cids = [c["chunk_id"] for c in ma_state.get("retrieved_chunks", [])]
            ma_answer = ma_state.get("final_answer", "")
            ma_cited_cids = ma_state.get("citations", [])
            ma_conflicts = ma_state.get("contradictions_found", [])

            ma_prec, ma_rec = compute_retrieval_metrics(ma_retrieved_cids, gt_chunks)
            ma_faith = compute_faithfulness(ma_cited_cids, ma_retrieved_cids, ma_answer)
            ma_contra = compute_contradiction_accuracy(ma_conflicts, has_conflict, gov_doc)

            # Sum tokens across all recorded agent node traces
            ma_tokens = sum([t.get("tokens", 0) for t in ma_state.get("trace_logs", [])])

            multi_agent_scores["precision"].append(ma_prec)
            multi_agent_scores["recall"].append(ma_rec)
            multi_agent_scores["faithfulness"].append(ma_faith)
            multi_agent_scores["contradiction_accuracy"].append(ma_contra)
            multi_agent_scores["latency_ms"].append(ma_latency)
            multi_agent_scores["tokens"].append(ma_tokens)

            print(f"  Naive RAG   -> Recall: {naive_rec:.2f} | Faithfulness: {naive_faith:.2f} | Contradiction: {naive_contra:.2f}")
            print(f"  Multi-Agent -> Recall: {ma_rec:.2f} | Faithfulness: {ma_faith:.2f} | Contradiction: {ma_contra:.2f}")

            detailed_results.append({
                "question_id": qid,
                "naive": {"recall": naive_rec, "faithfulness": naive_faith, "contradiction": naive_contra},
                "multi_agent": {"recall": ma_rec, "faithfulness": ma_faith, "contradiction": ma_contra}
            })

        # Calculate macro averages
        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

        summary = {
            "total_questions": len(self.eval_data),
            "naive_rag": {
                "avg_retrieval_precision": avg(naive_scores["precision"]),
                "avg_retrieval_recall": avg(naive_scores["recall"]),
                "avg_faithfulness": avg(naive_scores["faithfulness"]),
                "contradiction_accuracy": avg(naive_scores["contradiction_accuracy"]),
                "avg_latency_ms": int(avg(naive_scores["latency_ms"])),
                "avg_tokens": int(avg(naive_scores["tokens"]))
            },
            "multi_agent": {
                "avg_retrieval_precision": avg(multi_agent_scores["precision"]),
                "avg_retrieval_recall": avg(multi_agent_scores["recall"]),
                "avg_faithfulness": avg(multi_agent_scores["faithfulness"]),
                "contradiction_accuracy": avg(multi_agent_scores["contradiction_accuracy"]),
                "avg_latency_ms": int(avg(multi_agent_scores["latency_ms"])),
                "avg_tokens": int(avg(multi_agent_scores["tokens"]))
            },
            "detailed": detailed_results
        }

        return summary

    def format_comparison_table(self, summary: Dict[str, Any]) -> str:
        """
        Renders a formatted Markdown benchmark table comparing Naive RAG vs Multi-Agent.
        """
        n = summary["naive_rag"]
        m = summary["multi_agent"]

        table = (
            "| Metric | Naive Single-Shot RAG | Multi-Agent LangGraph System | Delta / Advantage |\n"
            "|---|---|---|---|\n"
            f"| **Retrieval Recall** | {n['avg_retrieval_recall'] * 100:.1f}% | **{m['avg_retrieval_recall'] * 100:.1f}%** | +{(m['avg_retrieval_recall'] - n['avg_retrieval_recall']) * 100:.1f}% (Query decomposition + Hybrid RRF) |\n"
            f"| **Retrieval Precision** | {n['avg_retrieval_precision'] * 100:.1f}% | **{m['avg_retrieval_precision'] * 100:.1f}%** | +{(m['avg_retrieval_precision'] - n['avg_retrieval_precision']) * 100:.1f}% (Deduplication + BM25 filtering) |\n"
            f"| **Faithfulness / Grounding** | {n['avg_faithfulness'] * 100:.1f}% | **{m['avg_faithfulness'] * 100:.1f}%** | +{(m['avg_faithfulness'] - n['avg_faithfulness']) * 100:.1f}% (Verification Critic gatekeeper) |\n"
            f"| **Contradiction Detection** | {n['contradiction_accuracy'] * 100:.1f}% | **{m['contradiction_accuracy'] * 100:.1f}%** | +{(m['contradiction_accuracy'] - n['contradiction_accuracy']) * 100:.1f}% (Explicit version comparison & precedence) |\n"
            f"| **Avg Latency per Query** | {n['avg_latency_ms']} ms | {m['avg_latency_ms']} ms | Multi-node graph orchestration overhead |\n"
            f"| **Avg Tokens per Query** | {n['avg_tokens']} tokens | {m['avg_tokens']} tokens | Multi-agent reasoning passes |"
        )
        return table
