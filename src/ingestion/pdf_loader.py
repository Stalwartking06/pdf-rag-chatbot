"""
PDF Loader module.

Responsible for extracting text from PDF files using pypdf, preserving
page numbers, handling empty/missing pages, and recording source metadata.
"""

from pathlib import Path
from typing import Union
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: Union[str, Path]) -> list[dict]:
    """
    Load a PDF file and extract text page by page.

    Args:
        pdf_path: Path to the PDF file (string or Path object).

    Returns:
        List of dictionaries with keys:
            - 'page_number' (int, 1-based)
            - 'text' (str, extracted text)
            - 'source' (str, file path)

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF file is empty or cannot be read.
    """
    path = Path(pdf_path)

    if not path.exists():
        logger.error(f"PDF file not found at: {path}")
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        logger.error(f"Path is not a file: {path}")
        raise ValueError(f"Specified path is not a file: {path}")

    logger.info(f"Loading PDF from: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.error(f"Failed to open PDF {path}: {e}")
        raise ValueError(f"Could not read PDF file: {e}") from e

    total_pages = len(reader.pages)
    if total_pages == 0:
        logger.warning(f"PDF file has 0 pages: {path}")
        return []

    pages_data: list[dict] = []
    total_chars = 0
    empty_pages = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text() or ""
        except Exception as page_err:
            logger.warning(f"Error extracting text from page {page_idx}: {page_err}")
            extracted = ""

        page_text = extracted.strip()
        if not page_text:
            empty_pages += 1

        total_chars += len(page_text)
        pages_data.append({
            "page_number": page_idx,
            "text": page_text,
            "source": str(path)
        })

    logger.info(
        f"PDF loaded: {total_pages} pages, {total_chars} total characters "
        f"({empty_pages} empty pages)."
    )
    return pages_data
