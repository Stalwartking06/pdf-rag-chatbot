"""
Evaluation Runner module.

Executes Information Retrieval evaluation across multiple test queries
derived from the company policy document. Computes Precision@5, Recall@5,
and Mean Reciprocal Rank (MRR) using ground-truth matching.
"""

from typing import Optional
import logging
from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank, mean_reciprocal_rank
from src.retrieval.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)

# Standard benchmark evaluation questions representing diverse policy topics
EVALUATION_QUERIES = [
    {
        "id": "Q1",
        "question": "What is the company mission?",
        "topic": "Company Mission",
        "target_phrases": ["mission statement", "our mission is to continue to lead the field of nurse triage"]
    },
    {
        "id": "Q2",
        "question": "Who is the CEO of the company?",
        "topic": "Leadership (CEO)",
        "target_phrases": ["Charu Raheja", "Chair and CEO"]
    },
    {
        "id": "Q3",
        "question": "Who is the COO of the company?",
        "topic": "Leadership (COO)",
        "target_phrases": ["Ravi Raheja", "Medical Director and COO"]
    },
    {
        "id": "Q4",
        "question": "What is the CEO salary?",
        "topic": "CEO Salary & Confidentiality",
        "target_phrases": ["Salaries and Rates of Pay", "strictly confidential", "Individual employees salary/wages are not public information"]
    },
    {
        "id": "Q5",
        "question": "What is the COO salary?",
        "topic": "COO Salary & Confidentiality",
        "target_phrases": ["Salaries and Rates of Pay", "strictly confidential", "Individual employees salary/wages are not public information"]
    },
    {
        "id": "Q6",
        "question": "What are the names and titles of managers in the company?",
        "topic": "Manager Names & Titles",
        "target_phrases": ["Nurse Managers", "Client Relations Manager", "Sales Account Manager", "Marketing Manager", "IT relations manager"]
    },
    {
        "id": "Q7",
        "question": "What are the responsibilities of employees?",
        "topic": "Employee Responsibilities",
        "target_phrases": ["Employee Responsibility", "satisfactorily complete their assigned responsibilities", "personal conduct and comments in the workplace"]
    },
    {
        "id": "Q8",
        "question": "What is the company leave policy?",
        "topic": "Leave Policy",
        "target_phrases": ["call in sick, use leave", "appropriate personnel procedures", "suspension without leave"]
    },
    {
        "id": "Q9",
        "question": "What happens if an employee takes sick leave or calls in sick?",
        "topic": "Sick Leave Procedures",
        "target_phrases": ["call in sick, use leave", "appropriate personnel procedures"]
    },
    {
        "id": "Q10",
        "question": "What is the vacation policy and how many leaves can employees take?",
        "topic": "Vacation Policy & Quantities",
        "target_phrases": ["call in sick, use leave", "Employee Classifications"]
    },
    {
        "id": "Q11",
        "question": "What employee benefits are provided?",
        "topic": "Employee Benefits",
        "target_phrases": ["For purposes of salary administration, eligibility for overtime payments, and employee benefits", "benefit statements"]
    }
]


def resolve_ground_truth(
    queries: list[dict],
    chunks: list[str],
    metadata: list[dict]
) -> list[set[int]]:
    """
    Derive ground-truth relevant chunk IDs by searching for target phrases
    in the document chunks. Ensures ground truth matches the actual chunking.

    Args:
        queries: List of query dictionaries with 'target_phrases'.
        chunks: List of chunk text strings.
        metadata: List of chunk metadata dictionaries.

    Returns:
        List of sets of relevant chunk IDs (one set per query).
    """
    ground_truth_sets: list[set[int]] = []

    for q in queries:
        target_phrases = q.get("target_phrases", [])
        relevant_ids: set[int] = set()

        for idx, chunk_text in enumerate(chunks):
            chunk_normalized = " ".join(chunk_text.lower().split())
            for phrase in target_phrases:
                phrase_normalized = " ".join(phrase.lower().split())
                if phrase_normalized in chunk_normalized:
                    relevant_ids.add(idx)
                    break

        ground_truth_sets.append(relevant_ids)

    return ground_truth_sets


def run_evaluation(
    retriever: HybridRetriever,
    chunks: list[str],
    metadata: list[dict],
    k: int = 5,
    custom_queries: Optional[list[dict]] = None
) -> dict:
    """
    Run retrieval evaluation across test queries and compute metrics.

    Args:
        retriever: HybridRetriever instance.
        chunks: All document chunk texts.
        metadata: All chunk metadata records.
        k: Evaluation rank cutoff (default 5).
        custom_queries: Optional custom query list.

    Returns:
        Dictionary with per-query results and aggregate metrics.
    """
    queries = custom_queries if custom_queries is not None else EVALUATION_QUERIES
    ground_truth_sets = resolve_ground_truth(queries, chunks, metadata)

    per_query_results = []
    all_retrieved_ids = []

    print("\n" + "=" * 70)
    print("RUNNING RETRIEVAL EVALUATION (Hybrid Search + RRF + Cross-Encoder)")
    print("=" * 70)

    for i, (q, relevant_set) in enumerate(zip(queries, ground_truth_sets), start=1):
        question = q["question"]
        topic = q.get("topic", f"Query {i}")

        # Retrieve chunks using the production hybrid pipeline
        retrieved_candidates = retriever.retrieve(
            query=question,
            top_k=k,
            candidate_k=20,
            rrf_k=60,
            rerank=True
        )

        retrieved_ids = [item["chunk_id"] for item in retrieved_candidates]
        all_retrieved_ids.append(retrieved_ids)

        p_k = precision_at_k(retrieved_ids, relevant_set, k=k)
        r_k = recall_at_k(retrieved_ids, relevant_set, k=k)
        rr = reciprocal_rank(retrieved_ids, relevant_set)

        result_row = {
            "query_id": q.get("id", f"Q{i}"),
            "topic": topic,
            "question": question,
            "relevant_chunks_count": len(relevant_set),
            "relevant_chunk_ids": sorted(list(relevant_set)),
            "retrieved_chunk_ids": retrieved_ids,
            "precision_at_k": p_k,
            "recall_at_k": r_k,
            "reciprocal_rank": rr
        }
        per_query_results.append(result_row)

        print(f"\n[{result_row['query_id']}] Topic: {topic}")
        print(f"  Question: \"{question}\"")
        print(f"  Relevant Chunks ({len(relevant_set)}): {result_row['relevant_chunk_ids']}")
        print(f"  Retrieved Top-{k}: {retrieved_ids}")
        print(f"  Precision@{k}: {p_k:.4f} | Recall@{k}: {r_k:.4f} | RR: {rr:.4f}")

    # Compute aggregates
    total_queries = len(queries)
    avg_precision = (
        sum(r["precision_at_k"] for r in per_query_results) / total_queries
        if total_queries > 0 else 0.0
    )
    avg_recall = (
        sum(r["recall_at_k"] for r in per_query_results) / total_queries
        if total_queries > 0 else 0.0
    )
    mrr = mean_reciprocal_rank(all_retrieved_ids, ground_truth_sets)

    summary = {
        "k": k,
        "total_queries": total_queries,
        "mean_precision_at_k": avg_precision,
        "mean_recall_at_k": avg_recall,
        "mean_reciprocal_rank": mrr,
        "per_query_results": per_query_results
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Queries      : {total_queries}")
    print(f"Precision@{k} (Mean)     : {avg_precision:.4f}")
    print(f"Recall@{k} (Mean)        : {avg_recall:.4f}")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print("=" * 70)

    return summary
