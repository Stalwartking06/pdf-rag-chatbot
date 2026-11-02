"""
Unit tests for PDF loading and extraction.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.ingestion.pdf_loader import load_pdf


def test_missing_pdf_handling(tmp_path: Path):
    """Test that a missing PDF raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        load_pdf(non_existent)


def test_invalid_path_handling(tmp_path: Path):
    """Test that providing a directory instead of a file raises ValueError."""
    with pytest.raises(ValueError):
        load_pdf(tmp_path)


def test_empty_pdf_handling(tmp_path: Path):
    """Test handling of a PDF with 0 pages."""
    fake_pdf = tmp_path / "empty.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF")

    with patch("src.ingestion.pdf_loader.PdfReader") as mock_reader:
        mock_instance = MagicMock()
        mock_instance.pages = []
        mock_reader.return_value = mock_instance

        pages = load_pdf(fake_pdf)
        assert pages == []


def test_pdf_extraction_page_metadata(tmp_path: Path):
    """Test successful text extraction and page metadata preservation."""
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF")

    with patch("src.ingestion.pdf_loader.PdfReader") as mock_reader:
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 Content: Welcome to Company Policy."

        page2 = MagicMock()
        page2.extract_text.return_value = ""  # Empty page

        page3 = MagicMock()
        page3.extract_text.return_value = "Page 3 Content: Mission Statement."

        mock_instance = MagicMock()
        mock_instance.pages = [page1, page2, page3]
        mock_reader.return_value = mock_instance

        pages = load_pdf(fake_pdf)

        assert len(pages) == 3
        assert pages[0]["page_number"] == 1
        assert "Welcome" in pages[0]["text"]
        assert pages[0]["source"] == str(fake_pdf)

        assert pages[1]["page_number"] == 2
        assert pages[1]["text"] == ""

        assert pages[2]["page_number"] == 3
        assert "Mission" in pages[2]["text"]
