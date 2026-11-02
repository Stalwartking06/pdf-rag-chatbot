"""
Text Chunking module.

Breaks down extracted page texts into smaller, overlapping chunks suitable
for embedding and retrieval, while preserving page number and source metadata.
"""

from typing import Union
import logging

logger = logging.getLogger(__name__)


def find_sensible_split(text: str, start: int, target_end: int, min_chunk: int = 50) -> int:
    """
    Find a natural boundary (paragraph break, sentence ending, or whitespace)
    near target_end to avoid cutting words or sentences in half.

    Args:
        text: Full text string.
        start: Starting index of the chunk.
        target_end: Desired ending index of the chunk.
        min_chunk: Minimum length to consider a valid boundary split.

    Returns:
        Integer index for the end of the chunk.
    """
    if target_end >= len(text):
        return len(text)

    # Use adaptive min_chunk so short chunks can still find boundaries
    effective_min = min(min_chunk, (target_end - start) // 3)
    search_start = max(start + effective_min, target_end - 150)
    window = text[search_start:target_end]

    # 1. Try paragraph break
    para_idx = window.rfind("\n\n")
    if para_idx != -1:
        return search_start + para_idx + 2

    # 2. Try sentence ending (period followed by space or newline)
    for punct in [". ", "? ", "! ", ".\n"]:
        sent_idx = window.rfind(punct)
        if sent_idx != -1:
            return search_start + sent_idx + len(punct)

    # 3. Try line break
    newline_idx = window.rfind("\n")
    if newline_idx != -1:
        return search_start + newline_idx + 1

    # 4. Try whitespace (space)
    space_idx = window.rfind(" ")
    if space_idx != -1:
        return search_start + space_idx + 1

    # Fallback to target_end
    return target_end


def chunk_text(
    pages_data: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> tuple[list[str], list[dict]]:
    """
    Split pages into chunks with specified size and overlap.

    Args:
        pages_data: List of page dictionaries containing 'page_number', 'text', 'source'.
        chunk_size: Maximum characters per chunk (must be > chunk_overlap).
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        Tuple of (chunks_list, metadata_list) where:
            - chunks_list: list of chunk text strings.
            - metadata_list: list of dictionaries with 'chunk_id', 'page_number', 'source'.

    Raises:
        ValueError: If chunk_size <= chunk_overlap or chunk_size <= 0.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be strictly greater than "
            f"chunk_overlap ({chunk_overlap})"
        )

    chunks: list[str] = []
    chunk_metadata: list[dict] = []

    for page in pages_data:
        page_number = page.get("page_number", 0)
        page_text = page.get("text", "")
        source = page.get("source", "unknown")

        if not page_text:
            continue

        start = 0
        text_len = len(page_text)

        while start < text_len:
            # Desired raw end
            target_end = min(start + chunk_size, text_len)

            # Prefer sensible text boundaries where practical
            end = find_sensible_split(page_text, start, target_end)

            chunk = page_text[start:end].strip()

            if chunk:
                chunk_id = len(chunks)
                chunks.append(chunk)
                chunk_metadata.append({
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "source": source
                })

            if end >= text_len:
                break

            # Strictly advance start to avoid infinite loop
            next_start = end - chunk_overlap
            start = max(start + 1, next_start)

    logger.info(
        f"Created {len(chunks)} chunks from {len(pages_data)} pages "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks, chunk_metadata
