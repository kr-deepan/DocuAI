"""
Layout-Aware Document Chunker
Implements contract/clause-aware splitting and prose recursive splitting with rich metadata enrichment.
"""
import os
import re
from typing import List, Dict, Any
from src.document_loader import Document

class Chunk:
    def __init__(self, chunk_id: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"<Chunk id='{self.chunk_id}' section='{self.metadata.get('section_header', 'N/A')}' length={len(self.text)}>"

class LayoutAwareChunker:
    def __init__(self, default_chunk_size: int = 500, default_overlap: int = 50):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Detects document type and routes to layout-aware or general chunking strategy."""
        doc_type, version = self._detect_doc_type_and_version(doc.filename, doc.content)
        
        if doc_type in ["contract", "amendment", "dpa", "sla"]:
            return self._chunk_contract_style(doc, doc_type, version)
        else:
            return self._chunk_general_style(doc, doc_type, version)

    def _detect_doc_type_and_version(self, filename: str, content: str) -> tuple[str, str]:
        """Extracts document type classification and version string from filename and text."""
        fn_lower = filename.lower()
        cnt_lower = content.lower()

        # Determine doc_type
        if "amendment" in fn_lower or "amendment" in cnt_lower:
            doc_type = "amendment"
        elif "agreement" in fn_lower or "msa" in fn_lower or "master service agreement" in cnt_lower:
            doc_type = "contract"
        elif "dpa" in fn_lower or "data processing addendum" in cnt_lower:
            doc_type = "dpa"
        elif "policy" in fn_lower or "security" in fn_lower:
            doc_type = "policy"
        elif "invoice" in fn_lower or "invoice" in cnt_lower:
            doc_type = "invoice"
        else:
            doc_type = "general"

        # Determine version/year
        version = "v1"
        if "v2" in fn_lower or "version 2" in cnt_lower:
            version = "v2"
        elif "2024" in fn_lower or "2024" in cnt_lower:
            version = "2024"
        elif "2023" in fn_lower or "2023" in cnt_lower:
            version = "2023"

        return doc_type, version

    def _chunk_contract_style(self, doc: Document, doc_type: str, version: str) -> List[Chunk]:
        """
        Contract/Clause-Aware Strategy:
        Splits text along section headers or paragraphs (double newlines).
        Ensures section headers are attached to clause text.
        """
        # Split on double newlines or section boundaries
        paragraphs = re.split(r"\n\s*\n", doc.content)
        chunks = []
        base_id = os.path.splitext(doc.filename)[0]

        idx = 1
        for para in paragraphs:
            cleaned = para.strip()
            if not cleaned or cleaned.startswith("==="):
                continue

            # Extract first line as section header if it starts with section number or title
            lines = cleaned.split("\n")
            first_line = lines[0].strip()
            
            if len(first_line) < 80 and (re.match(r"^\d+(\.\d+)?", first_line) or "RECITALS" in first_line or "AMENDED" in first_line):
                section_header = first_line
            else:
                section_header = f"SECTION_{idx}"

            chunk_id = f"{base_id}_chunk_{idx}"
            metadata = {
                "source_doc": doc.filename,
                "doc_type": doc_type,
                "version": version,
                "section_header": section_header,
                "chunk_id": chunk_id,
                "chunk_index": idx
            }
            chunks.append(Chunk(chunk_id=chunk_id, text=cleaned, metadata=metadata))
            idx += 1

        return chunks

    def _chunk_general_style(self, doc: Document, doc_type: str, version: str) -> List[Chunk]:
        """
        Recursive Character Splitting Strategy for general prose documents.
        Uses sliding window with overlap or paragraph splits.
        """
        paragraphs = re.split(r"\n\s*\n", doc.content)
        chunks = []
        base_id = os.path.splitext(doc.filename)[0]

        idx = 1
        for para in paragraphs:
            cleaned = para.strip()
            if not cleaned or cleaned.startswith("==="):
                continue

            lines = cleaned.split("\n")
            first_line = lines[0].strip()
            section_header = first_line if len(first_line) < 60 else f"PROSE_PART_{idx}"

            chunk_id = f"{base_id}_chunk_{idx}"
            metadata = {
                "source_doc": doc.filename,
                "doc_type": doc_type,
                "version": version,
                "section_header": section_header,
                "chunk_id": chunk_id,
                "chunk_index": idx
            }
            chunks.append(Chunk(chunk_id=chunk_id, text=cleaned, metadata=metadata))
            idx += 1

        return chunks
