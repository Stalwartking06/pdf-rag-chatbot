"""
Embedding Model module.

Wraps SentenceTransformer for generating normalized float32 dense embeddings
with dimension validation and model instance reuse.
"""

from typing import Optional, Union
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Manages loading and inference for the dense embedding model.
    Reuses the model instance across calls to avoid redundant memory allocations.
    """

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the SentenceTransformer model.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        # Verify embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model '{model_name}' loaded successfully. "
            f"Dimension: {self.dimension}"
        )

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingModel":
        """Singleton accessor to prevent reloading the model repeatedly."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode a list of texts into normalized float32 vectors.

        Args:
            texts: Single text or list of texts to embed.
            batch_size: Batch size for encoding.
            normalize: Whether to normalize vectors to unit length (L2 norm).
            show_progress_bar: Display progress bar during encoding.

        Returns:
            2D numpy array of shape (num_texts, dimension) with float32 dtype.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        # Validate dimension
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        return embeddings
