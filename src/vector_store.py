"""
Persistent ChromaDB Vector Store Wrapper
Supports dynamic multi-project isolation via dedicated ChromaDB collections.
"""
import os
import re
from typing import List, Dict, Any, Optional
import chromadb

from src.config import CHROMA_PERSIST_DIR
from src.chunker import Chunk

class ChromaVectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

    def _sanitize_collection_name(self, project_id: str) -> str:
        """
        ChromaDB collection names must be 3-63 characters, start/end with alphanum,
        and contain only [a-zA-Z0-9_-].
        """
        if not project_id or project_id == "default":
            return "contradiction_aware_docs"
        
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", project_id.strip().lower())
        col_name = f"proj_{cleaned}"
        # Ensure length constraint
        if len(col_name) > 63:
            col_name = col_name[:63]
        if len(col_name) < 3:
            col_name = f"{col_name}_idx"
        return col_name

    def get_collection(self, project_id: str = "default"):
        """Retrieves or creates the isolated collection for a specific project."""
        col_name = self._sanitize_collection_name(project_id)
        return self.client.get_or_create_collection(
            name=col_name,
            metadata={"project_id": project_id, "description": f"Isolated chunks for project '{project_id}'"}
        )

    @property
    def collection(self):
        """Backward-compatible default collection property."""
        return self.get_collection("default")

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        project_id: str = "default"
    ) -> None:
        """Upserts chunks and embeddings into a project's isolated collection."""
        if not chunks or not embeddings:
            return

        col = self.get_collection(project_id)
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        
        # Ensure project_id is tagged into chunk metadata
        metadatas = []
        for c in chunks:
            meta = c.metadata.copy() if c.metadata else {}
            meta["project_id"] = project_id
            metadatas.append(meta)

        col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"[VECTOR STORE] Successfully indexed {len(chunks)} chunks in ChromaDB ('{col.name}', project='{project_id}')")

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        project_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """Performs vector similarity search against a project's isolated collection."""
        col = self.get_collection(project_id)
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        retrieved = []
        if results and results.get("ids") and results["ids"][0]:
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

    def get_count(self, project_id: str = "default") -> int:
        """Returns total chunk count in a project's collection."""
        col = self.get_collection(project_id)
        return col.count()

    def get_all_chunks(self, project_id: str = "default") -> Dict[str, Any]:
        """Returns all stored ids, documents, and metadatas for a project."""
        col = self.get_collection(project_id)
        return col.get()

    def reset(self, project_id: str = "default") -> None:
        """Resets collection for a specific project."""
        col_name = self._sanitize_collection_name(project_id)
        try:
            self.client.delete_collection(col_name)
        except Exception:
            pass
        self.get_collection(project_id)

    def delete_project_collection(self, project_id: str) -> bool:
        """Permanently deletes a project's collection from ChromaDB."""
        if project_id == "default":
            return False
        col_name = self._sanitize_collection_name(project_id)
        try:
            self.client.delete_collection(col_name)
            return True
        except Exception as e:
            print(f"[VECTOR STORE WARNING] Could not delete collection '{col_name}': {e}")
            return False

# Singleton instance
vector_store = ChromaVectorStore()
