"""
Evaluation Metrics module.

Contains standard Information Retrieval (IR) evaluation metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

These metrics evaluate the quality of the retrieval stage independently
of the language model generation stage.
"""

from typing import Union, Sequence, Set


def precision_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Union[Set[int], Sequence[int]],
    k: int = 5
) -> float:
    """
    Calculate Precision@K: the proportion of retrieved documents in the top-K
    that are truly relevant.

    Formula:
        Precision@K = |{retrieved in top-K} ∩ {relevant}| / K

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Set or list of known ground-truth relevant IDs.
        k: Cutoff rank (must be > 0).

    Returns:
        Float score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_set = set(relevant_ids)
    relevant_retrieved = sum(1 for cid in top_k if cid in relevant_set)

    return relevant_retrieved / float(k)


def recall_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Union[Set[int], Sequence[int]],
    k: int = 5
) -> float:
    """
    Calculate Recall@K: the proportion of total relevant documents that
    were successfully retrieved in the top-K.

    Formula:
        Recall@K = |{retrieved in top-K} ∩ {relevant}| / |{total relevant}|

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Set or list of known ground-truth relevant IDs.
        k: Cutoff rank (must be > 0).

    Returns:
        Float score between 0.0 and 1.0 (or 0.0 if no relevant items exist).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for cid in top_k if cid in relevant_set)

    return relevant_retrieved / float(len(relevant_set))


def reciprocal_rank(
    retrieved_ids: Sequence[int],
    relevant_ids: Union[Set[int], Sequence[int]]
) -> float:
    """
    Calculate Reciprocal Rank (RR): 1 divided by the 1-based rank of the FIRST
    relevant document retrieved.

    Formula:
        RR = 1 / rank_first_relevant (or 0.0 if no relevant item is found).

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Set or list of known ground-truth relevant IDs.

    Returns:
        Float score between 0.0 and 1.0.
    """
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: Sequence[Sequence[int]],
    relevant_sets: Sequence[Union[Set[int], Sequence[int]]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across a collection of queries.

    Formula:
        MRR = (1 / |Q|) * sum( RR_q for q in Q )

    Args:
        retrieved_lists: List of retrieved chunk ID lists (one per query).
        relevant_sets: List of ground-truth relevant ID sets (one per query).

    Returns:
        Mean Reciprocal Rank float score between 0.0 and 1.0.
    """
    if not retrieved_lists or not relevant_sets:
        return 0.0

    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError("Number of queries must match number of relevant sets.")

    total_rr = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_lists, relevant_sets)
    )

    return total_rr / float(len(retrieved_lists))
