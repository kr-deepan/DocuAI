"""
Ingestion Pipeline Coordinator
Processes raw documents, generates layout-aware chunks, embeds via OpenRouter,
indexes into ChromaDB, and persists metadata and extracted text in SQLite.
Supports both the default pre-indexed legal corpus and dynamic, isolated multi-project workspaces.
"""
import os
import glob
import uuid
from typing import List, Optional

from src.config import EMBEDDING_MODEL
from src.core.model_checker import validate_configured_models
from src.core.openrouter_client import openrouter_client
from src.core.database import db_manager
from src.document_loader import DocumentLoader, Document
from src.chunker import LayoutAwareChunker, Chunk
from src.vector_store import vector_store
from src.retrieval.hybrid_search import get_hybrid_retriever

class IngestionPipeline:
    def __init__(self, sample_dir: str = os.path.join("data", "sample_docs")):
        self.sample_dir = sample_dir
        self.loader = DocumentLoader()
        self.chunker = LayoutAwareChunker()

    def run(self, reset_existing: bool = True, project_id: str = "default") -> int:
        """Executes full document ingestion pipeline for a directory of documents into a project."""
        print("=" * 60)
        print(f"Starting Document Ingestion Pipeline for Project: '{project_id}'")
        print("=" * 60)

        # 1. Validate free models
        validate_configured_models()

        if reset_existing:
            if project_id == "default":
                vector_store.reset()
            else:
                vector_store.delete_project_collection(project_id)

        # 2. Collect sample files
        supported_extensions = ["*.txt", "*.pdf", "*.png", "*.jpg", "*.jpeg"]
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(glob.glob(os.path.join(self.sample_dir, ext)))

        if not file_paths:
            print(f"[INGESTION] No sample documents found in '{self.sample_dir}'")
            return 0

        print(f"[INGESTION] Found {len(file_paths)} sample documents to ingest into project '{project_id}'.")

        all_chunks: List[Chunk] = []

        # 3. Load & Chunk documents
        for file_path in file_paths:
            doc = self.loader.load_file(file_path)
            chunks = self.chunker.chunk_document(doc)
            
            # Tag each chunk with project_id
            for c in chunks:
                c.metadata["project_id"] = project_id

            all_chunks.extend(chunks)

            # Persist document to SQLite database
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else len(doc.content.encode("utf-8"))
            doc_id = f"{project_id}_{doc.filename}"
            db_manager.save_document(
                doc_id=doc_id,
                project_id=project_id,
                filename=doc.filename,
                content=doc.content,
                file_type=doc.metadata.get("file_type", "txt"),
                file_size=file_size,
                chunk_count=len(chunks)
            )

            print(f"  Processed '{doc.filename}': {len(chunks)} chunks persisted to DB and queued for indexing.")

        print(f"[INGESTION] Total generated chunks across corpus: {len(all_chunks)}")

        # 4. Generate Embeddings via OpenRouter
        texts_to_embed = [c.text for c in all_chunks]
        print(f"[INGESTION] Embedding {len(texts_to_embed)} chunks via OpenRouter model '{EMBEDDING_MODEL}'...")
        embeddings = openrouter_client.embed_texts(EMBEDDING_MODEL, texts_to_embed, batch_size=16)

        # 5. Index into project-scoped ChromaDB collection
        vector_store.add_chunks(all_chunks, embeddings, project_id=project_id)

        # 6. Refresh Hybrid BM25 Index for this project
        retriever = get_hybrid_retriever(project_id)
        retriever.refresh_index()

        total_count = vector_store.get_count(project_id=project_id)
        print("-" * 60)
        print(f"Ingestion Complete! Project '{project_id}' ChromaDB index size: {total_count} chunks.")
        print("-" * 60)
        return total_count


def ingest_project_document(
    project_id: str,
    filename: str,
    content: str,
    file_type: str = "txt",
    file_size: int = 0
) -> int:
    """
    Ingests a single text document into a specific project workspace.
    Chunks text, embeds chunks, stores in ChromaDB, records document in SQLite,
    and refreshes the project's BM25 index.
    """
    if not project_id:
        project_id = "default"

    # Ensure project exists in SQLite
    proj = db_manager.get_project(project_id)
    if not proj:
        db_manager.create_project(
            project_id=project_id,
            name=project_id.replace("_", " ").title(),
            description="Auto-created project workspace"
        )

    # 1. Wrap in Document structure
    doc = Document(
        filename=filename,
        content=content,
        metadata={
            "source_doc": filename,
            "file_type": file_type,
            "project_id": project_id
        }
    )

    # 2. Chunk document
    chunker = LayoutAwareChunker()
    chunks = chunker.chunk_document(doc)
    if not chunks:
        # Fallback for short single paragraphs
        chunks = [
            Chunk(
                chunk_id=f"{os.path.splitext(filename)[0]}_chunk_1",
                text=content,
                metadata={
                    "source_doc": filename,
                    "doc_type": "general",
                    "version": "v1",
                    "section_header": "GENERAL",
                    "chunk_id": f"{os.path.splitext(filename)[0]}_chunk_1",
                    "chunk_index": 1,
                    "project_id": project_id
                }
            )
        ]

    for c in chunks:
        c.metadata["project_id"] = project_id
        # Namespace chunk ID to guarantee uniqueness in vector index
        if not c.chunk_id.startswith(f"{project_id}_"):
            c.chunk_id = f"{project_id}_{c.chunk_id}"
            c.metadata["chunk_id"] = c.chunk_id

    # 3. Generate embeddings
    texts_to_embed = [c.text for c in chunks]
    embeddings = openrouter_client.embed_texts(EMBEDDING_MODEL, texts_to_embed, batch_size=16)

    # 4. Upsert into project collection
    vector_store.add_chunks(chunks, embeddings, project_id=project_id)

    # 5. Save document in SQLite
    doc_id = f"{project_id}_{uuid.uuid4().hex[:6]}_{filename}"
    db_manager.save_document(
        doc_id=doc_id,
        project_id=project_id,
        filename=filename,
        content=content,
        file_type=file_type,
        file_size=file_size if file_size > 0 else len(content.encode("utf-8")),
        chunk_count=len(chunks)
    )

    # 6. Refresh Hybrid Retriever BM25 index
    retriever = get_hybrid_retriever(project_id)
    retriever.refresh_index()

    return len(chunks)


def ingest_project_file_bytes(
    project_id: str,
    file_bytes: bytes,
    filename: str
) -> int:
    """
    Ingests an uploaded file (from Streamlit or FastAPI) by parsing its raw bytes
    (handling TXT, PDF, OCR image extraction) and storing it into the project workspace.
    """
    loader = DocumentLoader()
    doc = loader.load_bytes(file_bytes, filename)
    file_type = doc.metadata.get("file_type", "txt")
    file_size = len(file_bytes)
    return ingest_project_document(
        project_id=project_id,
        filename=filename,
        content=doc.content,
        file_type=file_type,
        file_size=file_size
    )


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
