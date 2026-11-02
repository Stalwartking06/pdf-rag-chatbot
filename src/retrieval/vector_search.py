"""
FAISS Vector Search module.

Implements FAISS IndexFlatIP for exact maximum inner product search
(cosine similarity with normalized vectors) with optional metadata filtering.
"""

from typing import Optional
import logging
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSSearch:
    """
    FAISS-based vector index for fast similarity search.
    Uses IndexFlatIP so that dot product of normalized vectors equals cosine similarity.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize the FAISS index.

        Args:
            dimension: Dimensionality of the vectors (e.g. 384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []
        self.raw_embeddings: Optional[np.ndarray] = None

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors stored in the index."""
        return self.index.ntotal

    def add_documents(
        self,
        embeddings: np.ndarray,
        metadata: list[dict]
    ) -> None:
        """
        Add document embeddings and their corresponding metadata to the index.

        Args:
            embeddings: 2D numpy array of float32 embeddings.
            metadata: List of metadata dictionaries (matching embeddings length).
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not match "
                f"metadata length ({len(metadata)})"
            )

        if len(embeddings) == 0:
            logger.warning("Empty embeddings array provided to FAISS index.")
            return

        embeddings_f32 = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings_f32)

        self.index.add(embeddings_f32)
        self.metadata.extend(metadata)
        self.raw_embeddings = (
            embeddings_f32
            if self.raw_embeddings is None
            else np.vstack([self.raw_embeddings, embeddings_f32])
        )

        logger.info(
            f"Added {len(embeddings)} vectors to FAISS index. "
            f"Total vectors: {self.total_vectors}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        page_filter: Optional[int] = None
    ) -> list[dict]:
        """
        Search for top-K most similar chunks to the query embedding.

        Args:
            query_embedding: 1D or 2D numpy array of the query vector.
            top_k: Number of top results to retrieve.
            page_filter: Optional page number to restrict search to.

        Returns:
            List of dictionaries:
                - 'chunk_id': int
                - 'score': float (cosine similarity between -1.0 and 1.0)
                - 'metadata': dict
        """
        if self.total_vectors == 0:
            logger.warning("Search called on an empty FAISS index.")
            return []

        q_vec = np.asarray(query_embedding, dtype="float32")
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        faiss.normalize_L2(q_vec)

        # Handle page filtering if requested
        if page_filter is not None:
            filtered_indices = [
                i for i, meta in enumerate(self.metadata)
                if meta.get("page_number") == page_filter
            ]

            if not filtered_indices:
                logger.info(f"No chunks found for page {page_filter}.")
                return []

            if self.raw_embeddings is None:
                logger.warning("Raw embeddings not retained for filtering.")
                return []

            filtered_embeddings = self.raw_embeddings[filtered_indices].copy()
            temp_index = faiss.IndexFlatIP(self.dimension)
            temp_index.add(filtered_embeddings)

            k_search = min(top_k, len(filtered_indices))
            scores, indices = temp_index.search(q_vec, k_search)

            results = []
            for score, local_idx in zip(scores[0], indices[0]):
                if local_idx < 0:
                    continue
                orig_idx = filtered_indices[local_idx]
                results.append({
                    "chunk_id": self.metadata[orig_idx].get("chunk_id", orig_idx),
                    "score": float(score),
                    "metadata": self.metadata[orig_idx]
                })
            return results

        # Unfiltered global search
        k_search = min(top_k, self.total_vectors)
        scores, indices = self.index.search(q_vec, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "chunk_id": self.metadata[idx].get("chunk_id", idx),
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results
