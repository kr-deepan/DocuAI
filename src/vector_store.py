"""
Persistent ChromaDB Vector Store Wrapper
Manages local Chroma vector collection and metadata querying.
"""
import os
from typing import List, Dict, Any, Optional
import chromadb

from src.config import CHROMA_PERSIST_DIR
from src.chunker import Chunk

class ChromaVectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = "contradiction_aware_docs"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks with contradiction-aware metadata"}
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Upserts chunks and their embeddings into ChromaDB collection."""
        if not chunks or not embeddings:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"[VECTOR STORE] Successfully indexed {len(chunks)} chunks in ChromaDB ('{self.collection_name}')")

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Performs vector similarity search against ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        retrieved = []
        if results and results.get("ids"):
            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(ids)):
                retrieved.append({
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "metadata": metadatas[i],
                    "distance": distances[i] if i < len(distances) else None
                })
        return retrieved

    def get_count(self) -> int:
        """Returns total chunk count in collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Resets collection for fresh ingestion tests."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

# Singleton instance
vector_store = ChromaVectorStore()
