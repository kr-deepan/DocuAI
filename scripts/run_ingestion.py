"""
CLI script to run the Phase 1 document ingestion pipeline.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import IngestionPipeline

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run(reset_existing=True)
