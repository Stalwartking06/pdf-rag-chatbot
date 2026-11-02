"""
Unit tests for retrieval modules: FAISS, BM25, RRF, and Cross-Encoder reranker.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from src.retrieval.vector_search import FAISSSearch
from src.retrieval.bm25_search import BM25Search
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_search import reciprocal_rank_fusion, HybridRetriever


def test_faiss_indexing_and_top_k():
    """Test FAISS indexing and top-k retrieval."""
    dim = 4
    searcher = FAISSSearch(dimension=dim)

    # 3 dummy vectors
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype="float32")

    metadata = [
        {"chunk_id": 0, "page_number": 1},
        {"chunk_id": 1, "page_number": 2},
        {"chunk_id": 2, "page_number": 3}
    ]

    searcher.add_documents(vectors, metadata)
    assert searcher.total_vectors == 3

    # Query identical to vector 0
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    results = searcher.search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == 0
    assert np.isclose(results[0]["score"], 1.0, atol=1e-5)


def test_faiss_metadata_filtering():
    """Test FAISS search with page_filter applied."""
    dim = 4
    searcher = FAISSSearch(dimension=dim)

    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype="float32")

    metadata = [
        {"chunk_id": 0, "page_number": 1},
        {"chunk_id": 1, "page_number": 5},
        {"chunk_id": 2, "page_number": 5}
    ]

    searcher.add_documents(vectors, metadata)

    # Search with page_filter=5: should ONLY return chunks from page 5
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    results = searcher.search(query, top_k=5, page_filter=5)

    assert len(results) == 2
    for r in results:
        assert r["metadata"]["page_number"] == 5
    assert results[0]["chunk_id"] == 1


def test_faiss_empty_results():
    """Test FAISS search on an empty index returns empty list."""
    searcher = FAISSSearch(dimension=4)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    results = searcher.search(query, top_k=5)
    assert results == []


def test_bm25_retrieval_and_stop_words():
    """Test BM25 retrieval for matching keywords."""
    chunks = [
        "The company mission is to provide excellent nurse triage services.",
        "Employee vacation policy and leave guidelines are explained here.",
        "Technical architecture and database server infrastructure."
    ]
    metadata = [
        {"chunk_id": 0, "page_number": 1},
        {"chunk_id": 1, "page_number": 2},
        {"chunk_id": 2, "page_number": 3}
    ]

    bm25 = BM25Search(chunks, metadata)

    results = bm25.search("nurse triage mission", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == 0
    assert results[0]["score"] > 0


def test_bm25_empty_query():
    """Test BM25 handles empty and whitespace-only queries."""
    chunks = ["Some text here."]
    metadata = [{"chunk_id": 0, "page_number": 1}]
    bm25 = BM25Search(chunks, metadata)

    assert bm25.search("") == []
    assert bm25.search("   ") == []


def test_rrf_duplicate_removal_and_ranking():
    """Test Reciprocal Rank Fusion removes duplicates and ranks correctly."""
    # List 1: [10, 20, 30]
    # List 2: [20, 10, 40]
    ranked_lists = [
        [10, 20, 30],
        [20, 10, 40]
    ]
    fused = reciprocal_rank_fusion(ranked_lists, k=60)

    # All chunk IDs in fused results must be unique
    fused_ids = [cid for cid, score in fused]
    assert len(fused_ids) == len(set(fused_ids))
    assert set(fused_ids) == {10, 20, 30, 40}

    # Items present in both lists (10 and 20) must score higher than single-list items (30 and 40)
    score_10 = dict(fused)[10]
    score_30 = dict(fused)[30]
    assert score_10 > score_30


def test_cross_encoder_reranker():
    """Test CrossEncoder reranker scores and sorts candidates descending."""
    mock_ce_model = MagicMock()
    # Mock predict returning higher score for chunk 1 than chunk 0
    mock_ce_model.predict.return_value = np.array([-1.2, 3.8])

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = mock_ce_model
    reranker.model_name = "mock-ce"

    candidates = [
        {"chunk_id": 0, "metadata": {"page_number": 1}},
        {"chunk_id": 1, "metadata": {"page_number": 2}}
    ]
    all_chunks = ["General text", "Specific mission text"]

    reranked = reranker.rerank("What is the mission?", candidates, all_chunks, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == 1  # Score 3.8 ranked higher
    assert reranked[1]["chunk_id"] == 0


def test_hybrid_retriever_full_flow():
    """Test end-to-end flow of HybridRetriever."""
    # Mock components
    mock_vector = MagicMock()
    mock_vector.search.return_value = [
        {"chunk_id": 0, "score": 0.8, "metadata": {"page_number": 1}},
        {"chunk_id": 1, "score": 0.6, "metadata": {"page_number": 2}}
    ]

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        {"chunk_id": 1, "score": 5.0, "metadata": {"page_number": 2}},
        {"chunk_id": 2, "score": 3.0, "metadata": {"page_number": 3}}
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"chunk_id": 1, "score": 2.5, "metadata": {"page_number": 2}, "text": "Chunk 1 text"}
    ]

    mock_emb = MagicMock()
    mock_emb.encode.return_value = np.zeros((1, 4), dtype="float32")

    retriever = HybridRetriever(
        vector_search=mock_vector,
        bm25_search=mock_bm25,
        reranker=mock_reranker,
        embedding_model=mock_emb,
        all_chunks=["Text 0", "Text 1", "Text 2"],
        metadata=[{"page_number": 1}, {"page_number": 2}, {"page_number": 3}]
    )

    results = retriever.retrieve("What is the policy?", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == 1
