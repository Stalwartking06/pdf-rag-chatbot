"""
Main Application Entry Point for the PDF RAG Chatbot.

Provides commands to start the interactive chatbot CLI, run a single query,
or execute the Information Retrieval evaluation framework.

Usage:
    python main.py                  # Start interactive chatbot CLI
    python main.py --eval           # Run IR retrieval evaluation benchmark
    python main.py --query "..."    # Run a single query and print result
"""

import argparse
import sys
import logging
from pathlib import Path

from src.config import (
    DEFAULT_PDF_PATH,
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    setup_logging
)
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import chunk_text
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_search import FAISSSearch
from src.retrieval.bm25_search import BM25Search
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.answer_generator import AnswerGenerator
from src.chatbot.chat import RAGChatbot
from src.evaluation.evaluation_runner import run_evaluation

logger = logging.getLogger("pdf_rag_app")


def build_rag_pipeline(pdf_path: Path = DEFAULT_PDF_PATH):
    """
    Initialize and assemble all components of the RAG pipeline.

    Args:
        pdf_path: Path to the target PDF document.

    Returns:
        Tuple of (chatbot, retriever, chunks, metadata)
    """
    print(f"Loading document from: {pdf_path}")
    pages_data = load_pdf(pdf_path)

    print(f"Chunking document (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks, metadata = chunk_text(
        pages_data,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if not chunks:
        raise ValueError(f"No text chunks could be extracted from {pdf_path}")

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedding_model = EmbeddingModel.get_instance(EMBEDDING_MODEL_NAME)

    print(f"Generating dense embeddings for {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)

    print("Building FAISS vector index...")
    vector_search = FAISSSearch(dimension=embeddings.shape[1])
    vector_search.add_documents(embeddings, metadata)

    print("Building BM25 keyword index...")
    bm25_search = BM25Search(chunks, metadata)

    print(f"Loading Cross-Encoder model ({CROSS_ENCODER_MODEL_NAME})...")
    reranker = CrossEncoderReranker.get_instance(CROSS_ENCODER_MODEL_NAME)

    print("Assembling Hybrid Retriever...")
    retriever = HybridRetriever(
        vector_search=vector_search,
        bm25_search=bm25_search,
        reranker=reranker,
        embedding_model=embedding_model,
        all_chunks=chunks,
        metadata=metadata
    )

    print(f"Connecting to Ollama ({LLM_MODEL_NAME})...")
    answer_generator = AnswerGenerator(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE
    )

    chatbot = RAGChatbot(
        retriever=retriever,
        answer_generator=answer_generator,
        chunks=chunks,
        metadata=metadata
    )

    return chatbot, retriever, chunks, metadata


def main():
    """Main CLI handler."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="PDF-based RAG Chatbot using FAISS, BM25, Cross-Encoder, and Ollama."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(DEFAULT_PDF_PATH),
        help="Path to the PDF document."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the Information Retrieval evaluation framework."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and output the answer."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed retrieval telemetry (candidates, RRF, Cross-Encoder scores)."
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        print(f"Error: Target PDF not found at {pdf_path}")
        sys.exit(1)

    chatbot, retriever, chunks, metadata = build_rag_pipeline(pdf_path)

    if args.eval:
        print("\nRunning Evaluation Benchmark...")
        run_evaluation(retriever, chunks, metadata, k=5)
        return

    if args.query:
        print(f"\nQuestion: {args.query}")
        result = chatbot.ask(args.query, debug=args.debug)

        if args.debug and result.get("debug_info"):
            dbg = result["debug_info"]
            print("\n--- [RETRIEVAL DEBUG TELEMETRY] ---")
            print(f"Original Query      : {dbg['original_query']}")
            print(f"Expanded Query      : {dbg['expanded_query']}")
            print(f"FAISS Candidates    : {dbg['faiss_candidates_count']}")
            print(f"BM25 Candidates     : {dbg['bm25_candidates_count']}")
            print(f"Hybrid Candidates   : {dbg['hybrid_candidates_count']}")
            if dbg["cross_encoder_scores"]:
                print("Cross-Encoder Candidate Scores:")
                for s in dbg["cross_encoder_scores"]:
                    print(f"  Chunk {s['chunk_id']} (Page {s['page_number']}): {s['score']:.4f}")
            print("-----------------------------------")

        print("\nAnswer:")
        print(result["answer"])
        if result["retrieved_chunks"]:
            print("\nSources:")
            for idx, c in enumerate(result["retrieved_chunks"], 1):
                pno = c.get("metadata", {}).get("page_number", "?")
                cid = c.get("chunk_id", "?")
                score = c.get("score", 0.0)
                print(f"  [{idx}] Page {pno} (Chunk {cid}, Score: {score:.4f})")
        return

    # Start interactive chat CLI
    chatbot.start_cli(debug_mode=args.debug)


if __name__ == "__main__":
    main()