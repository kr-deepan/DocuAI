"""
Phase 1 Verification Script
Validates budget guard, SQLite response caching, document loading, layout-aware chunking, and ChromaDB retrieval.
"""
import os
import sys
import sqlite3

# Ensure backend root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import EMBEDDING_MODEL, RESPONSE_CACHE_PATH
from src.core.budget import budget_guard
from src.core.cache import response_cache
from src.core.openrouter_client import openrouter_client
from src.ingestion import IngestionPipeline
from src.vector_store import vector_store

def verify_phase1():
    print("=" * 60)
    print("Running Phase 1 Verification Checks")
    print("=" * 60)

    # Clean response cache table for fresh verification
    if os.path.exists(RESPONSE_CACHE_PATH):
        with sqlite3.connect(RESPONSE_CACHE_PATH) as conn:
            conn.execute("DELETE FROM response_cache")
            conn.commit()

    # 1. Test Budget Guard & Cache functionality
    initial_usage = budget_guard.get_usage()
    print(f"[CHECK 1] Initial daily budget usage: {initial_usage}/50 requests used today.")

    test_payload = ["Unit test text for cache verification"]
    
    # First embedding call (populates cache & increments budget if not cached)
    embeddings1 = openrouter_client.embed_texts(EMBEDDING_MODEL, test_payload)
    usage_after_first = budget_guard.get_usage()
    print(f"[CHECK 2] Usage after 1st embedding call: {usage_after_first}")

    # Second identical embedding call (MUST HIT CACHE & NOT INCREMENT BUDGET)
    embeddings2 = openrouter_client.embed_texts(EMBEDDING_MODEL, test_payload)
    usage_after_second = budget_guard.get_usage()
    print(f"[CHECK 3] Usage after 2nd (cached) embedding call: {usage_after_second}")

    assert usage_after_first == usage_after_second, "FAILED: Cache hit incremented budget quota!"
    print("[PASS] Response cache successfully prevented duplicate network request & quota usage.")

    # 2. Run Document Ingestion Pipeline
    print("\n[CHECK 4] Running Ingestion Pipeline...")
    pipeline = IngestionPipeline()
    chunk_count = pipeline.run(reset_existing=True)
    
    assert chunk_count >= 10, f"FAILED: Ingestion produced only {chunk_count} chunks (expected >= 10)."
    print(f"[PASS] Ingestion indexed {chunk_count} chunks in ChromaDB.")

    # 3. Test Retrieval Query
    query = "termination notice period 30 days 90 days"
    print(f"\n[CHECK 5] Executing vector similarity search for query: '{query}'...")
    query_emb = openrouter_client.embed_texts(EMBEDDING_MODEL, [query])[0]
    
    retrieved = vector_store.similarity_search(query_emb, top_k=5)
    print(f"Retrieved {len(retrieved)} relevant chunks:")
    
    found_v1 = False
    found_v2 = False
    
    for idx, res in enumerate(retrieved):
        meta = res["metadata"]
        src = meta.get("source_doc")
        sec = meta.get("section_header")
        print(f"  Result {idx+1}: [{src}] Section: '{sec}' (Distance: {res['distance']:.4f})")
        if "master_service_agreement_v1" in src:
            found_v1 = True
        if "msa_amendment_v2" in src:
            found_v2 = True

    assert found_v1 and found_v2, "FAILED: Retrieval did not return chunks from both MSA v1 and MSA v2!"
    print("[PASS] Vector search successfully retrieved matching chunks from both MSA v1 and MSA v2!")

    print("=" * 60)
    print("Phase 1 Verification: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    verify_phase1()
