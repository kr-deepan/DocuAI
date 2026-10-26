"""
Trace Logger for Multi-Agent Observability

Role in the System:
-------------------
Provides a lightweight, file-persisted JSON Lines trace log recording every
agent state transition without external cloud dependencies (e.g., LangSmith).

Recorded Fields per Agent Step:
-------------------------------
- timestamp: ISO-8601 UTC timestamp
- agent_name: Name of the agent node (e.g., 'orchestrator', 'retriever', 'synthesis')
- input_summary: Compact representation of inputs received by the node
- output_summary: Compact representation of decisions/text produced
- tokens: Token usage consumed by any LLM call inside this step
- latency_ms: Execution duration of the step in milliseconds
- metadata: Optional additional attributes (e.g. sub-query list, model name)

Why a custom logger rather than LangSmith:
-----------------------------------------
For a portfolio project, understanding how to construct your own observability
pipeline demonstrates fundamental understanding of tracing, structured logging,
and debugging agentic state machines.
"""
import os
import json
import datetime
from typing import Dict, Any, Optional

from src.config import BASE_DIR

LOGS_DIR = os.path.join(BASE_DIR, "logs")
TRACE_LOG_PATH = os.path.join(LOGS_DIR, "agent_traces.jsonl")

class TraceLogger:
    def __init__(self, log_path: str = TRACE_LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_step(
        self,
        agent_name: str,
        input_summary: Any,
        output_summary: Any,
        tokens: int = 0,
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Appends an agent trace entry to logs/agent_traces.jsonl and returns the dict
        so nodes can easily append it to the shared LangGraph state.
        """
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "agent_name": agent_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[TRACE LOGGER WARNING] Failed to write trace log: {e}")

        return entry

    def get_recent_traces(self, limit: int = 20) -> list[Dict[str, Any]]:
        """Reads recent trace entries from the log file."""
        if not os.path.exists(self.log_path):
            return []
        
        traces = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return traces[-limit:]

# Singleton instance
trace_logger = TraceLogger()
