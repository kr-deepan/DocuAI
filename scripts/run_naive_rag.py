"""
CLI Runner for Naive Single-Shot RAG Baseline

Usage:
------
python scripts/run_naive_rag.py "What is the payment term under the agreement?"
python scripts/run_naive_rag.py --interactive
"""
import sys
import os
import argparse

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.naive_rag import naive_rag

def print_result(result: dict):
    print("\n" + "=" * 70)
    print(f"QUERY: {result['query']}")
    print("=" * 70)
    print(f"\nANSWER:\n{result['answer']}\n")
    print("-" * 70)
    print(f"RETRIEVED CHUNKS ({len(result['retrieved_chunks'])} chunks):")
    for idx, c in enumerate(result['retrieved_chunks'], 1):
        meta = c.get("metadata", {})
        print(f"  [{idx}] {meta.get('source_doc', 'Unknown')} - {meta.get('section_header', 'General')} (dist: {c.get('distance', 0):.4f})")
    print("-" * 70)
    usage = result.get("usage", {})
    print(f"METRICS: Latency: {result['latency_ms']}ms | Tokens: {usage.get('total_tokens', 0)} (prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}) | Model: {result.get('model_used')}")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run Naive RAG Baseline Query")
    parser.add_argument("query", nargs="?", type=str, help="Question to ask Naive RAG")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive prompt mode")
    args = parser.parse_args()

    if args.interactive:
        print("=== Naive Single-Shot RAG Interactive Mode (Type 'exit' to quit) ===")
        while True:
            try:
                user_q = input("\nEnter Question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    break
                res = naive_rag.query(user_q)
                print_result(res)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    elif args.query:
        res = naive_rag.query(args.query)
        print_result(res)
    else:
        # Default test query demonstrating contradiction
        default_q = "What is the termination notice period under the agreement?"
        print(f"[INFO] No query provided. Running default contradiction test query: '{default_q}'")
        res = naive_rag.query(default_q)
        print_result(res)

if __name__ == "__main__":
    main()
