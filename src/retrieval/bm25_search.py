"""
BM25 Keyword Search module.

Implements lexical keyword search using rank-bm25 (BM25Okapi).
Complements dense vector retrieval by excelling at exact terms, acronyms, and names.
"""

from typing import Optional
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from src.utils.text_utils import tokenize

logger = logging.getLogger(__name__)


class BM25Search:
    """
    BM25 keyword search index.
    Tokenizes document chunks and computes Okapi BM25 relevance scores.
    """

    def __init__(self, chunks: list[str], metadata: list[dict]) -> None:
        """
        Initialize and index document chunks with BM25.

        Args:
            chunks: List of document chunk text strings.
            metadata: List of metadata dictionaries corresponding to each chunk.
        """
        if len(chunks) != len(metadata):
            raise ValueError("Chunks and metadata lists must have equal length.")

        self.chunks = chunks
        self.metadata = metadata
        self.corpus_size = len(chunks)

        if self.corpus_size > 0:
            self.tokenized_corpus = [
                tokenize(chunk, remove_stopwords=False)
                for chunk in chunks
            ]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Initialized BM25 index with {self.corpus_size} chunks.")
        else:
            self.tokenized_corpus = []
            self.bm25 = None
            logger.warning("BM25 initialized with an empty corpus.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        page_filter: Optional[int] = None
    ) -> list[dict]:
        """
        Retrieve top-K matching chunks using BM25 scoring.

        Args:
            query: User search query.
            top_k: Number of top results to return.
            page_filter: Optional page number to filter chunks.

        Returns:
            List of dictionaries:
                - 'chunk_id': int
                - 'score': float (BM25 score >= 0)
                - 'metadata': dict
        """
        if not query or not query.strip() or self.bm25 is None or self.corpus_size == 0:
            return []

        # Tokenize query, filtering out noise words
        query_tokens = tokenize(query, remove_stopwords=True)
        if not query_tokens:
            # Fallback to unfiltered tokens if all were stopwords
            query_tokens = tokenize(query, remove_stopwords=False)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Apply page filter if specified
        if page_filter is not None:
            valid_indices = [
                i for i, meta in enumerate(self.metadata)
                if meta.get("page_number") == page_filter and scores[i] > 0
            ]
        else:
            valid_indices = [i for i in range(self.corpus_size) if scores[i] > 0]

        if not valid_indices:
            return []

        # Sort valid indices by score descending
        sorted_indices = sorted(
            valid_indices,
            key=lambda idx: float(scores[idx]),
            reverse=True
        )[:top_k]

        results = []
        for idx in sorted_indices:
            results.append({
                "chunk_id": self.metadata[idx].get("chunk_id", idx),
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })

        return results
