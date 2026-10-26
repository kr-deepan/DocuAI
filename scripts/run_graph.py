"""
CLI Runner for LangGraph Multi-Agent System (Phase 3)

Usage:
------
python scripts/run_graph.py "Compare the termination notice between MSA v1 and Amendment v2"
python scripts/run_graph.py --interactive
"""
import sys
import os
import argparse

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import run_agent_system

def print_agent_trace(state: dict):
    print("\n" + "=" * 75)
    print(f"QUERY: {state['original_query']}")
    print("=" * 75)

    print("\n--- AGENT EXECUTION TRACE ---")
    traces = state.get("trace_logs", [])
    for idx, t in enumerate(traces, 1):
        name = t.get("agent_name", "").upper()
        latency = t.get("latency_ms", 0)
        tokens = t.get("tokens", 0)
        print(f"  Step {idx}: [{name}] ({latency}ms | {tokens} tokens)")
        out_summary = t.get("output_summary")
        if isinstance(out_summary, dict):
            for k, v in out_summary.items():
                print(f"    - {k}: {v}")
        else:
            print(f"    - output: {out_summary}")

    print("\n" + "-" * 75)
    print(f"RETRIEVED CHUNKS ({len(state.get('retrieved_chunks', []))} chunks):")
    for idx, c in enumerate(state.get("retrieved_chunks", []), 1):
        meta = c.get("metadata", {})
        print(f"  [{idx}] {c['chunk_id']} | {meta.get('source_doc')} ({meta.get('section_header')}) | RRF: {c.get('rrf_score', 0):.4f}")

    print("-" * 75)
    print("SYNTHESIZED ANSWER:")
    print(state.get("final_answer", ""))
    print("-" * 75)
    print(f"INLINE CITATIONS ({len(state.get('citations', []))} cited): {state.get('citations', [])}")
    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run LangGraph Multi-Agent Query")
    parser.add_argument("query", nargs="?", type=str, help="Question to ask the agent system")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt")
    args = parser.parse_args()

    if args.interactive:
        print("=== Multi-Agent System Interactive Mode (Type 'exit' to quit) ===")
        while True:
            try:
                user_q = input("\nEnter Question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    break
                state = run_agent_system(user_q)
                print_agent_trace(state)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    elif args.query:
        state = run_agent_system(args.query)
        print_agent_trace(state)
    else:
        default_q = "What is the termination notice period and payment terms under MSA v1 vs Amendment v2?"
        print(f"[INFO] No query provided. Running default query:\n'{default_q}'\n")
        state = run_agent_system(default_q)
        print_agent_trace(state)

if __name__ == "__main__":
    main()
