"""
Hybrid Retrieval Engine (Dense Vector + BM25 Keyword Search with Reciprocal Rank Fusion)

Role in the System:
-------------------
Combines semantic search (dense embeddings via ChromaDB) with exact lexical
keyword search (`rank_bm25`). Merges disparate ranking lists using
Reciprocal Rank Fusion (RRF).

Inputs & Outputs:
-----------------
- Input: Query text string, optional metadata filters (e.g. version="v2"), top_k
- Output: List of fused, ranked chunks with metadata and RRF relevance scores

Why Hybrid Search instead of Dense Vector Search alone:
-------------------------------------------------------
Dense vector embeddings understand conceptual meanings (e.g., "termination"
matches "canceling the agreement"), but struggle with exact numbers, legal section
citations (e.g., "Section 8.1"), or technical SKU identifiers. BM25 catches exact
lexical terms that embeddings blur.

Why Reciprocal Rank Fusion (RRF):
---------------------------------
Dense similarity outputs cosine/L2 distance scores (e.g., 0.2 to 1.8), while
BM25 outputs unbounded positive score integers (e.g., 2.4 to 18.7).
Directly adding or weighting raw scores requires complex normalization tuning.
RRF bypasses score calibration by using pure relative ranks:
    RRF_Score(doc) = 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)
"""
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.config import EMBEDDING_MODEL
from src.core.openrouter_client import openrouter_client
from src.vector_store import vector_store

def tokenize_text(text: str) -> List[str]:
    """Lightweight lowercase word tokenizer for BM25."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if len(w) > 1]

class HybridRetriever:
    def __init__(self, top_k_dense: int = 8, top_k_bm25: int = 8, rrf_k: int = 60):
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.rrf_k = rrf_k  # Standard RRF smoothing constant
        self.bm25_corpus: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Pulls all stored chunks from ChromaDB and compiles the BM25 index."""
        try:
            stored = vector_store.collection.get()
            if not stored or not stored.get("ids"):
                self.bm25_corpus = []
                self.bm25_index = None
                return

            ids = stored["ids"]
            docs = stored["documents"]
            metadatas = stored["metadatas"]

            corpus = []
            tokenized_corpus = []

            for i in range(len(ids)):
                item = {
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "metadata": metadatas[i]
                }
                corpus.append(item)
                tokenized_corpus.append(tokenize_text(docs[i]))

            self.bm25_corpus = corpus
            if tokenized_corpus:
                self.bm25_index = BM25Okapi(tokenized_corpus)
                print(f"[HYBRID RETRIEVER] BM25 index built with {len(corpus)} chunks.")
        except Exception as e:
            print(f"[HYBRID RETRIEVER WARNING] Could not build BM25 index: {e}")
            self.bm25_corpus = []
            self.bm25_index = None

    def refresh_index(self) -> None:
        """Refreshes the BM25 index (call after fresh document ingestion)."""
        self._build_bm25_index()

    def dense_search(
        self,
        query: str,
        top_k: int = 8,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Dense embedding similarity search against ChromaDB."""
        query_embs = openrouter_client.embed_texts(EMBEDDING_MODEL, [query])
        if not query_embs:
            return []
        
        query_emb = query_embs[0]
        results = vector_store.similarity_search(query_emb, top_k=top_k, where_filter=where_filter)
        return results

    def bm25_search(
        self,
        query: str,
        top_k: int = 8,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Sparse keyword search using BM25 with optional metadata filtering."""
        if not self.bm25_index or not self.bm25_corpus:
            self._build_bm25_index()
            if not self.bm25_index:
                return []

        tokenized_q = tokenize_text(query)
        if not tokenized_q:
            return []

        doc_scores = self.bm25_index.get_scores(tokenized_q)

        # Pair scores with corpus items
        scored_docs = []
        for idx, score in enumerate(doc_scores):
            item = self.bm25_corpus[idx]
            
            # Apply metadata filter if specified
            if where_filter:
                match = True
                item_meta = item.get("metadata", {})
                for k, v in where_filter.items():
                    if item_meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            scored_docs.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "bm25_score": float(score)
            })

        # Sort descending by BM25 score
        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_docs[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes both Dense and BM25 search, merges them via Reciprocal Rank Fusion,
        and returns the top_k fused results.
        """
        dense_results = self.dense_search(query, top_k=self.top_k_dense, where_filter=where_filter)
        bm25_results = self.bm25_search(query, top_k=self.top_k_bm25, where_filter=where_filter)

        # Track scores and metadata by chunk_id
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # 1. Accumulate Dense Ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            item["dense_rank"] = rank

        # 2. Accumulate BM25 Ranks
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            chunk_map[cid]["bm25_rank"] = rank

        # 3. Sort by total RRF score
        fused_items = []
        for cid, score in rrf_scores.items():
            entry = chunk_map[cid].copy()
            entry["rrf_score"] = score
            fused_items.append(entry)

        fused_items.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_items[:top_k]

# Singleton instance
hybrid_retriever = HybridRetriever()
