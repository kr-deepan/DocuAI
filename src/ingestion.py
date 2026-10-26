"""
Ingestion Pipeline Coordinator
Processes raw sample documents, generates layout-aware chunks, embeds via OpenRouter, and indexes into ChromaDB.
"""
import os
import glob
from typing import List

from src.config import EMBEDDING_MODEL
from src.core.model_checker import validate_configured_models
from src.core.openrouter_client import openrouter_client
from src.document_loader import DocumentLoader, Document
from src.chunker import LayoutAwareChunker, Chunk
from src.vector_store import vector_store

class IngestionPipeline:
    def __init__(self, sample_dir: str = os.path.join("data", "sample_docs")):
        self.sample_dir = sample_dir
        self.loader = DocumentLoader()
        self.chunker = LayoutAwareChunker()

    def run(self, reset_existing: bool = True) -> int:
        """Executes full document ingestion pipeline."""
        print("=" * 60)
        print("Starting Document Ingestion Pipeline (Phase 1)")
        print("=" * 60)

        # 1. Validate free models
        validate_configured_models()

        if reset_existing:
            vector_store.reset()

        # 2. Collect sample files
        supported_extensions = ["*.txt", "*.pdf", "*.png", "*.jpg", "*.jpeg"]
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(glob.glob(os.path.join(self.sample_dir, ext)))

        if not file_paths:
            print(f"[INGESTION] No sample documents found in '{self.sample_dir}'")
            return 0

        print(f"[INGESTION] Found {len(file_paths)} sample documents to ingest.")

        all_chunks: List[Chunk] = []

        # 3. Load & Chunk documents
        for file_path in file_paths:
            doc = self.loader.load_file(file_path)
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  Processed '{doc.filename}': generated {len(chunks)} chunks (type: {chunks[0].metadata['doc_type']})")

        print(f"[INGESTION] Total generated chunks across corpus: {len(all_chunks)}")

        # 4. Generate Embeddings via OpenRouter (Cached & Budget-Guarded)
        texts_to_embed = [c.text for c in all_chunks]
        print(f"[INGESTION] Embedding {len(texts_to_embed)} chunks via OpenRouter model '{EMBEDDING_MODEL}'...")
        embeddings = openrouter_client.embed_texts(EMBEDDING_MODEL, texts_to_embed, batch_size=16)

        # 5. Index into ChromaDB
        vector_store.add_chunks(all_chunks, embeddings)

        print("-" * 60)
        print(f"Ingestion Complete! ChromaDB total index size: {vector_store.get_count()} chunks.")
        print("-" * 60)
        return vector_store.get_count()

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
