"""
Unit tests for the embedding model wrapper.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.embeddings.embedding_model import EmbeddingModel


def test_embedding_dimensions_and_dtype():
    """Test that embedding model returns float32 vectors of correct dimension."""
    with patch("src.embeddings.embedding_model.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Return mock normalized float32 vectors
        fake_vectors = np.random.randn(2, 384).astype("float32")
        fake_vectors /= np.linalg.norm(fake_vectors, axis=1, keepdims=True)
        mock_model.encode.return_value = fake_vectors
        mock_st.return_value = mock_model

        model = EmbeddingModel("all-MiniLM-L6-v2")
        vectors = model.encode(["hello world", "company policy"])

        assert vectors.shape == (2, 384)
        assert vectors.dtype == np.float32

        # Verify unit norm
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedding_model_singleton():
    """Test that get_instance reuses the same instance."""
    with patch("src.embeddings.embedding_model.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        inst1 = EmbeddingModel.get_instance("mock-model-a")
        inst2 = EmbeddingModel.get_instance("mock-model-a")
        assert inst1 is inst2


def test_empty_input_handling():
    """Test encoding an empty list returns empty array with correct shape."""
    with patch("src.embeddings.embedding_model.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        model = EmbeddingModel("all-MiniLM-L6-v2")
        vectors = model.encode([])
        assert vectors.shape == (0, 384)
        assert vectors.dtype == np.float32
