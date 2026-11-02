"""
Hybrid Search module.

Combines dense vector retrieval (FAISS) and sparse lexical retrieval (BM25)
using Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking.
"""

from typing import Optional
import logging
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_search import FAISSSearch
from src.retrieval.bm25_search import BM25Search
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.text_utils import expand_query

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists of chunk IDs using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(item) = sum( 1 / (k + rank) ) across all ranking systems.
        where rank is 1-indexed (1st place has rank=1).

    Args:
        ranked_lists: List of rankings, each ranking being a list of chunk IDs.
        k: Smoothing constant to balance high-rank dominance (standard default: 60).

    Returns:
        List of tuples (chunk_id, rrf_score) sorted descending by RRF score.
    """
    rrf_scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    # Sort descending by RRF score
    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
    return sorted_items


class HybridRetriever:
    """
    Coordinates Hybrid Retrieval across FAISS, BM25, RRF, and Cross-Encoder.
    """

    def __init__(
        self,
        vector_search: FAISSSearch,
        bm25_search: BM25Search,
        reranker: CrossEncoderReranker,
        embedding_model: EmbeddingModel,
        all_chunks: list[str],
        metadata: list[dict]
    ) -> None:
        """
        Initialize the Hybrid Retriever.

        Args:
            vector_search: FAISS vector index instance.
            bm25_search: BM25 keyword index instance.
            reranker: Cross-Encoder reranker instance.
            embedding_model: Embedding model instance.
            all_chunks: Complete list of chunk texts.
            metadata: Complete list of chunk metadata dictionaries.
        """
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.reranker = reranker
        self.embedding_model = embedding_model
        self.all_chunks = all_chunks
        self.metadata = metadata

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 20,
        rrf_k: int = 60,
        rerank: bool = True,
        threshold: Optional[float] = None,
        page_filter: Optional[int] = None,
        return_debug: bool = False
    ):
        """
        Perform complete hybrid retrieval for a query.

        Workflow:
            1. Expand user query with relevant domain synonyms.
            2. Dense vector search via FAISS.
            3. Lexical keyword search via BM25.
            4. Merge rankings using Reciprocal Rank Fusion (RRF).
            5. Deep semantic reranking using Cross-Encoder.

        Args:
            query: User question.
            top_k: Number of final chunks to return.
            candidate_k: Candidates retrieved from each searcher.
            rrf_k: Smoothing constant for RRF.
            rerank: Whether to apply Cross-Encoder reranking.
            threshold: Optional score cutoff for Cross-Encoder.
            page_filter: Optional page number filter.
            return_debug: Whether to return debug telemetry metadata.

        Returns:
            List of result dictionaries sorted by relevance,
            or (results, debug_info) tuple if return_debug=True.
        """
        if not query or not query.strip():
            if return_debug:
                return [], {}
            return []

        expanded_q = expand_query(query)

        # 1. FAISS Dense Retrieval
        query_emb = self.embedding_model.encode(expanded_q)
        vector_results = self.vector_search.search(
            query_emb,
            top_k=candidate_k,
            page_filter=page_filter
        )
        vector_chunk_ids = [res["chunk_id"] for res in vector_results]

        # 2. BM25 Lexical Retrieval
        bm25_results = self.bm25_search.search(
            expanded_q,
            top_k=candidate_k,
            page_filter=page_filter
        )
        bm25_chunk_ids = [res["chunk_id"] for res in bm25_results]

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [vector_chunk_ids, bm25_chunk_ids],
            k=rrf_k
        )

        if not fused:
            logger.info("No candidates returned from FAISS or BM25.")
            if return_debug:
                return [], {
                    "original_query": query,
                    "expanded_query": expanded_q,
                    "faiss_candidates_count": 0,
                    "bm25_candidates_count": 0,
                    "hybrid_candidates_count": 0,
                    "cross_encoder_scores": [],
                    "final_chunks": []
                }
            return []

        # Prepare candidate dictionary list for reranker
        candidate_ids = [chunk_id for chunk_id, _ in fused[:candidate_k]]
        candidates = [
            {
                "chunk_id": cid,
                "metadata": self.metadata[cid] if cid < len(self.metadata) else {},
                "rrf_score": score
            }
            for cid, score in fused[:candidate_k]
        ]

        # 4. Cross-Encoder Reranking
        reranked = []
        if rerank:
            reranked = self.reranker.rerank(
                query=expanded_q,
                candidates=candidates,
                all_chunks=self.all_chunks,
                top_k=top_k,
                threshold=threshold
            )
            if reranked:
                results = reranked
            else:
                logger.warning(
                    "Cross-Encoder produced no results after threshold filtering; "
                    "falling back to top RRF candidates."
                )
                results = []
                for cand in candidates[:top_k]:
                    cid = cand["chunk_id"]
                    results.append({
                        "chunk_id": cid,
                        "score": cand.get("rrf_score", 0.0),
                        "metadata": cand.get("metadata", {}),
                        "text": self.all_chunks[cid]
                    })
        else:
            results = []
            for cand in candidates[:top_k]:
                cid = cand["chunk_id"]
                results.append({
                    "chunk_id": cid,
                    "score": cand.get("rrf_score", 0.0),
                    "metadata": cand.get("metadata", {}),
                    "text": self.all_chunks[cid]
                })

        if return_debug:
            debug_info = {
                "original_query": query,
                "expanded_query": expanded_q,
                "faiss_candidates_count": len(vector_chunk_ids),
                "bm25_candidates_count": len(bm25_chunk_ids),
                "hybrid_candidates_count": len(fused),
                "cross_encoder_scores": [
                    {
                        "chunk_id": item["chunk_id"],
                        "score": round(item["score"], 4),
                        "page_number": item.get("metadata", {}).get("page_number", "N/A")
                    }
                    for item in reranked
                ],
                "final_chunks": [
                    {
                        "chunk_id": item["chunk_id"],
                        "score": round(item["score"], 4),
                        "page_number": item.get("metadata", {}).get("page_number", "N/A")
                    }
                    for item in results
                ]
            }
            return results, debug_info

        return results
