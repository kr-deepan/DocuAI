"""
Benchmark Runner Script (Phase 6)

Executes side-by-side evaluation of Naive Single-Shot RAG vs. Multi-Agent LangGraph,
prints the comparison table, and writes results to data/benchmark_results.json.

Usage:
------
python scripts/run_benchmark.py
python scripts/run_benchmark.py --update-readme
"""
import sys
import os
import json
import argparse

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import BASE_DIR
from src.eval.evaluator import BenchmarkEvaluator

README_PATH = os.path.join(BASE_DIR, "README.md")
BENCHMARK_RESULTS_PATH = os.path.join(BASE_DIR, "data", "benchmark_results.json")

def update_readme_table(markdown_table: str):
    """Replaces placeholder benchmark table in README.md with generated benchmark results."""
    if not os.path.exists(README_PATH):
        return

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = "## 📊 Benchmark Comparison (Naive RAG vs Multi-Agent)"
    end_marker = "## 📂 Project Structure"

    if start_marker in content and end_marker in content:
        before = content.split(start_marker)[0]
        after = content.split(end_marker)[1]
        new_section = (
            f"{start_marker}\n\n"
            f"Evaluated across the 6-question golden evaluation suite (`data/eval_set.json`):\n\n"
            f"{markdown_table}\n\n"
            f"---\n\n"
        )
        updated_content = f"{before}{new_section}{end_marker}{after}"
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print("[BENCHMARK] Successfully updated README.md with live benchmark table!")

def main():
    parser = argparse.ArgumentParser(description="Run RAG vs Multi-Agent Benchmark")
    parser.add_argument("--update-readme", action="store_true", help="Automatically update README.md with results table")
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator()
    summary = evaluator.evaluate()

    # Save JSON results
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[BENCHMARK] Full JSON results saved to: {BENCHMARK_RESULTS_PATH}")

    # Generate and print Markdown table
    markdown_table = evaluator.format_comparison_table(summary)
    print("\n" + "=" * 75)
    print("BENCHMARK COMPARISON TABLE (Naive Single-Shot RAG vs Multi-Agent)")
    print("=" * 75)
    print(markdown_table)
    print("=" * 75 + "\n")

    if args.update_readme:
        update_readme_table(markdown_table)

if __name__ == "__main__":
    main()
