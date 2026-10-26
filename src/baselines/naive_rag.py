"""
Naive Single-Shot RAG Pipeline (Baseline Benchmark)

Role in the System:
-------------------
This module implements a standard, linear Retrieval-Augmented Generation (RAG)
pipeline with ZERO multi-agent orchestration, NO query rewriting, NO hybrid
keyword search, NO verification critic, and NO contradiction detection.

Pipeline Architecture:
----------------------
[User Query] -> [Dense Embedding] -> [ChromaDB Vector Search (top-k)]
              -> [Prompt Stuffing] -> [LLM Synthesis] -> [Final Answer]

Inputs & Outputs:
-----------------
- Input: User query string (e.g., "What is the termination notice period?")
- Output: Dictionary containing:
    - query: original query string
    - answer: generated response from LLM
    - retrieved_chunks: list of raw chunks passed into the prompt context
    - latency_ms: total end-to-end execution latency in milliseconds
    - usage: token usage statistics (prompt, completion, total)

Why this is a separate module rather than folded into an agent:
--------------------------------------------------------------
In production system design and scientific evaluation, agentic architectures
must be quantitatively compared against a standard baseline. This module
serves as the permanent control group for Phase 6 benchmarking to measure
improvements in grounding, recall, and contradiction handling.
"""
import time
from typing import List, Dict, Any, Optional

from src.config import EMBEDDING_MODEL, SYNTHESIS_MODEL
from src.core.openrouter_client import openrouter_client
from src.vector_store import vector_store

class NaiveRAGPipeline:
    def __init__(
        self,
        top_k: int = 4,
        synthesis_model: str = SYNTHESIS_MODEL,
        embedding_model: str = EMBEDDING_MODEL
    ):
        self.top_k = top_k
        self.synthesis_model = synthesis_model
        self.embedding_model = embedding_model

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Embeds the query and retrieves the top-k nearest chunks from ChromaDB
        using standard dense vector similarity.
        """
        query_embeddings = openrouter_client.embed_texts(self.embedding_model, [query])
        if not query_embeddings:
            return []
        
        query_emb = query_embeddings[0]
        chunks = vector_store.similarity_search(query_emb, top_k=self.top_k)
        return chunks

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formats retrieved chunks into a linear context window and invokes
        the synthesis model to produce a single-shot response.
        """
        if not chunks:
            return {
                "content": "No relevant documents found to answer your question.",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": 0
            }

        # Build concatenated context string (standard prompt stuffing)
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source_doc", "Unknown")
            sec = meta.get("section_header", "General")
            context_parts.append(f"--- Document [{idx}]: {src} ({sec}) ---\n{chunk.get('text', '')}")

        context_str = "\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful assistant answering questions based strictly on the provided context documents. "
            "If the context does not contain the answer, say that you do not know. "
            "Do not assume or extrapolate information beyond what is directly stated."
        )

        user_prompt = (
            f"Context Documents:\n"
            f"{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = openrouter_client.chat_completion(
            model=self.synthesis_model,
            messages=messages,
            temperature=0.1,
            max_tokens=600
        )
        return response

    def query(self, query_str: str) -> Dict[str, Any]:
        """
        Executes end-to-end Naive RAG query flow and returns structured metrics.
        """
        start_time = time.time()

        # Step 1: Dense Retrieval
        retrieved_chunks = self.retrieve(query_str)

        # Step 2: Synthesis Generation
        generation_result = self.generate(query_str, retrieved_chunks)

        total_latency_ms = int((time.time() - start_time) * 1000)

        return {
            "query": query_str,
            "answer": generation_result.get("content", ""),
            "retrieved_chunks": retrieved_chunks,
            "latency_ms": total_latency_ms,
            "usage": generation_result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }),
            "model_used": self.synthesis_model
        }

# Singleton instance for quick access
naive_rag = NaiveRAGPipeline()
