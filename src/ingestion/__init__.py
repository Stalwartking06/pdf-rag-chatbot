"""Document ingestion package: PDF loading and chunking."""
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import chunk_text

__all__ = ["load_pdf", "chunk_text"]
