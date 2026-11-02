"""
Cross-Encoder Reranker module.

Reranks candidate chunks retrieved by hybrid search using a pretrained
Cross-Encoder model (ms-marco-MiniLM-L-6-v2) that processes the query and
passage together, capturing nuanced semantic interactions.
"""

from typing import Optional
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder model wrapper for passage reranking.
    Caches model weights to avoid repeated disk reads and GPU/CPU re-initialization.
    """

    _instance: Optional["CrossEncoderReranker"] = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Initialize the Cross-Encoder model.

        Args:
            model_name: Name of the Hugging Face cross-encoder model.
        """
        self.model_name = model_name
        logger.info(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)
        logger.info(f"Cross-Encoder model '{model_name}' loaded successfully.")

    @classmethod
    def get_instance(
        cls,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> "CrossEncoderReranker":
        """Singleton accessor to prevent reloading model weights repeatedly."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        all_chunks: list[str],
        top_k: int = 3,
        threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Score and rank candidate passages for the given query.

        Args:
            query: The search query text.
            candidates: List of candidate dictionaries containing 'chunk_id' and 'metadata'.
            all_chunks: The complete list of document chunk texts indexed by chunk_id.
            top_k: Maximum number of reranked candidates to return.
            threshold: Optional minimum score threshold. If None, no filtering is applied.

        Returns:
            List of reranked dictionaries sorted descending by score:
                - 'chunk_id': int
                - 'score': float (Cross-Encoder logit score)
                - 'metadata': dict
                - 'text': str (chunk text snippet)
        """
        if not query or not candidates:
            return []

        # Form pairs of (query, chunk_text)
        pairs: list[tuple[str, str]] = []
        valid_candidates: list[dict] = []

        for cand in candidates:
            cid = cand.get("chunk_id")
            if cid is not None and 0 <= cid < len(all_chunks):
                chunk_text = all_chunks[cid]
                pairs.append((query, chunk_text))
                valid_candidates.append(cand)

        if not pairs:
            logger.warning("No valid candidate text found for reranking.")
            return []

        scores = self.model.predict(pairs)

        # Build scored results
        scored_results: list[dict] = []
        for cand, score in zip(valid_candidates, scores):
            cid = cand["chunk_id"]
            float_score = float(score)

            # Apply threshold if specified
            if threshold is not None and float_score < threshold:
                continue

            scored_results.append({
                "chunk_id": cid,
                "score": float_score,
                "metadata": cand.get("metadata", {}),
                "text": all_chunks[cid]
            })

        # Sort descending by Cross-Encoder score
        scored_results.sort(key=lambda item: item["score"], reverse=True)

        return scored_results[:top_k]
