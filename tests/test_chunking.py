"""
Unit tests for text chunking and boundary handling.
"""

import pytest
from src.ingestion.chunking import chunk_text, find_sensible_split


def test_chunk_creation_and_overlap():
    """Test that text is split into chunks with proper overlap."""
    text = (
        "Sentence one is here. " * 30 +  # ~660 chars
        "\n\n" +
        "Paragraph two starts here. " * 30  # ~810 chars
    )
    pages = [{"page_number": 1, "text": text, "source": "test.pdf"}]

    chunks, metadata = chunk_text(pages, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 1
    assert len(chunks) == len(metadata)

    # Check metadata fields
    for i, meta in enumerate(metadata):
        assert meta["chunk_id"] == i
        assert meta["page_number"] == 1
        assert meta["source"] == "test.pdf"

    # Check non-empty chunks
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_overlap_shares_content():
    """Test that consecutive chunks share content due to overlap."""
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 " * 10
    pages = [{"page_number": 5, "text": text, "source": "doc.pdf"}]

    chunks, _ = chunk_text(pages, chunk_size=100, chunk_overlap=30)

    assert len(chunks) >= 2
    # Check that there is overlap between adjacent chunks
    chunk_0_tail = chunks[0][-20:]
    assert any(word in chunks[1] for word in chunk_0_tail.split())


def test_page_metadata_preservation():
    """Test that chunks preserve distinct page numbers across multi-page document."""
    pages = [
        {"page_number": 1, "text": "Content of page 1.", "source": "test.pdf"},
        {"page_number": 2, "text": "Content of page 2.", "source": "test.pdf"},
        {"page_number": 3, "text": "Content of page 3.", "source": "test.pdf"}
    ]

    chunks, metadata = chunk_text(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) == 3
    assert [m["page_number"] for m in metadata] == [1, 2, 3]


def test_avoid_empty_chunks():
    """Test that empty or blank pages do not produce chunks."""
    pages = [
        {"page_number": 1, "text": "", "source": "test.pdf"},
        {"page_number": 2, "text": "   \n\t  ", "source": "test.pdf"},
        {"page_number": 3, "text": "Valid content on page 3.", "source": "test.pdf"}
    ]

    chunks, metadata = chunk_text(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) == 1
    assert metadata[0]["page_number"] == 3


def test_chunking_parameter_validation():
    """Test validation of chunk_size and chunk_overlap parameters to prevent infinite loops."""
    pages = [{"page_number": 1, "text": "Some text", "source": "test.pdf"}]

    # chunk_overlap >= chunk_size
    with pytest.raises(ValueError):
        chunk_text(pages, chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        chunk_text(pages, chunk_size=100, chunk_overlap=150)

    # negative or zero chunk_size
    with pytest.raises(ValueError):
        chunk_text(pages, chunk_size=0, chunk_overlap=50)

    with pytest.raises(ValueError):
        chunk_text(pages, chunk_size=-10, chunk_overlap=0)


def test_sensible_split_finds_boundaries():
    """Test find_sensible_split locates paragraph and sentence endings."""
    text = "First paragraph here.\n\nSecond paragraph begins here with more words."
    split = find_sensible_split(text, start=0, target_end=35)
    # Target 35 should snap to paragraph break '\n\n'
    assert split == len("First paragraph here.\n\n")
