"""
Document Loader Module
Parses text files, PDFs via PyMuPDF (fitz), and images via pytesseract OCR fallback.
"""
import os
import logging
from typing import List, Dict, Any
import pymupdf  # PyMuPDF
from PIL import Image
import pytesseract

logger = logging.getLogger("DocumentLoader")

class Document:
    def __init__(self, filename: str, content: str, metadata: Dict[str, Any]):
        self.filename = filename
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"<Document filename='{self.filename}' pages={self.metadata.get('page_count', 1)} length={len(self.content)}>"

class DocumentLoader:
    def __init__(self, ocr_min_chars: int = 20):
        self.ocr_min_chars = ocr_min_chars

    def load_file(self, file_path: str) -> Document:
        """Loads a document file from path, applying OCR fallback if needed."""
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt":
            return self._load_txt(file_path, filename)
        elif ext == ".pdf":
            return self._load_pdf(file_path, filename)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            return self._load_image(file_path, filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _load_txt(self, file_path: str, filename: str) -> Document:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        metadata = {
            "source_doc": filename,
            "file_type": "txt",
            "page_count": 1
        }
        return Document(filename=filename, content=content, metadata=metadata)

    def _load_pdf(self, file_path: str, filename: str) -> Document:
        doc = pymupdf.open(file_path)
        full_text = []
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()

            # Apply OCR fallback if page has insufficient text
            if len(text.strip()) < self.ocr_min_chars:
                print(f"[OCR FALLBACK] Page {page_num + 1} of '{filename}' has insufficient text. Running Tesseract OCR...")
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    text = ocr_text if ocr_text.strip() else text
                except Exception as e:
                    logger.warning(f"OCR failed on page {page_num + 1}: {e}")

            full_text.append(text)

        doc.close()
        combined_content = "\n\n".join(full_text)
        metadata = {
            "source_doc": filename,
            "file_type": "pdf",
            "page_count": page_count
        }
        return Document(filename=filename, content=combined_content, metadata=metadata)

    def _load_image(self, file_path: str, filename: str) -> Document:
        print(f"[OCR EXTRACTION] Extracting text from image file '{filename}' via Tesseract...")
        img = Image.open(file_path)
        try:
            content = pytesseract.image_to_string(img)
        except Exception as e:
            logger.error(f"Failed to run Tesseract OCR on {filename}: {e}")
            content = f"[OCR Failed for {filename}]"
        
        metadata = {
            "source_doc": filename,
            "file_type": "image",
            "page_count": 1
        }
        return Document(filename=filename, content=content, metadata=metadata)
