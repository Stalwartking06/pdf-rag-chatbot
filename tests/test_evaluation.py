"""
Unit tests for Information Retrieval evaluation metrics: Precision@K, Recall@K, and MRR.
"""

import pytest
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank
)


def test_precision_at_k():
    """Test Precision@K calculation with various hits."""
    retrieved = [10, 20, 30, 40, 50]
    relevant = {20, 40}

    # In top-5: 2 hits (20, 40) out of 5 -> 2/5 = 0.40
    assert precision_at_k(retrieved, relevant, k=5) == 0.40

    # In top-2: 1 hit (20) out of 2 -> 1/2 = 0.50
    assert precision_at_k(retrieved, relevant, k=2) == 0.50

    # In top-1: 0 hits -> 0.0
    assert precision_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_at_k():
    """Test Recall@K calculation."""
    retrieved = [10, 20, 30, 40, 50]
    relevant = {20, 40, 99}  # 3 total relevant items

    # Top-5 retrieves 20 and 40 (2 out of 3 relevant) -> 2/3 ≈ 0.6667
    r5 = recall_at_k(retrieved, relevant, k=5)
    assert pytest.approx(r5, rel=1e-3) == 2.0 / 3.0

    # If all relevant retrieved in top-k -> 1.0
    assert recall_at_k(retrieved, {10, 30}, k=5) == 1.0

    # If no relevant items exist -> 0.0
    assert recall_at_k(retrieved, set(), k=5) == 0.0


def test_reciprocal_rank():
    """Test Reciprocal Rank for first relevant occurrence."""
    # First relevant at rank 1 -> 1/1 = 1.0
    assert reciprocal_rank([10, 20, 30], {10}) == 1.0

    # First relevant at rank 2 -> 1/2 = 0.5
    assert reciprocal_rank([10, 20, 30], {20}) == 0.5

    # First relevant at rank 3 -> 1/3 ≈ 0.3333
    assert pytest.approx(reciprocal_rank([10, 20, 30], {30}), rel=1e-3) == 1.0 / 3.0

    # Not found -> 0.0
    assert reciprocal_rank([10, 20, 30], {99}) == 0.0


def test_mean_reciprocal_rank():
    """Test Mean Reciprocal Rank (MRR) across multiple queries."""
    retrieved_lists = [
        [1, 2, 3],  # Hit at rank 1 -> RR = 1.0
        [1, 2, 3],  # Hit at rank 2 -> RR = 0.5
        [1, 2, 3]   # No hit -> RR = 0.0
    ]
    relevant_sets = [
        {1},
        {2},
        {99}
    ]

    # Mean = (1.0 + 0.5 + 0.0) / 3 = 1.5 / 3 = 0.5
    assert mean_reciprocal_rank(retrieved_lists, relevant_sets) == 0.5


def test_evaluation_invalid_k():
    """Test that k <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, k=0)

    with pytest.raises(ValueError):
        recall_at_k([1, 2], {1}, k=-1)
