"""
Interactive Chatbot module.

Provides the RAGChatbot orchestrator and CLI interface for end users,
handling spelling correction, role lookups, hybrid retrieval, and answer generation.
"""

from typing import Optional
import logging
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.answer_generator import AnswerGenerator
from src.utils.text_utils import (
    correct_spelling,
    build_vocabulary,
    is_vague_question,
    get_vague_question_suggestion,
    extract_company_roles
)

logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Main RAG Chatbot coordinator that wires together preprocessing,
    retrieval, and generative answer synthesis.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        answer_generator: AnswerGenerator,
        chunks: list[str],
        metadata: list[dict]
    ) -> None:
        """
        Initialize the RAG Chatbot.

        Args:
            retriever: HybridRetriever instance.
            answer_generator: AnswerGenerator instance.
            chunks: Complete list of document chunk texts.
            metadata: Complete list of chunk metadata records.
        """
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.chunks = chunks
        self.metadata = metadata
        self.vocabulary = build_vocabulary(chunks)
        logger.info(f"Chatbot initialized with vocabulary size: {len(self.vocabulary)}")

    def ask(self, user_question: str, debug: bool = False) -> dict:
        """
        Process a user question through the complete RAG pipeline.

        Args:
            user_question: Raw query string from user.
            debug: Whether to include retrieval debug telemetry.

        Returns:
            Dictionary containing:
                - 'answer': Final generated or lookup answer
                - 'original_question': Raw user input
                - 'corrected_question': Spell-corrected question
                - 'retrieved_chunks': List of retrieved chunks with metadata
                - 'retrieval_mode': 'exact_lookup', 'clarification', or 'hybrid_rag'
                - 'debug_info': Optional retrieval debug dictionary (if debug=True)
        """
        cleaned = user_question.strip()
        if not cleaned:
            return {
                "answer": "Please enter a valid question.",
                "original_question": user_question,
                "corrected_question": "",
                "retrieved_chunks": [],
                "retrieval_mode": "empty",
                "debug_info": None
            }

        # 1. Conservative Spelling Correction
        corrected = correct_spelling(cleaned, self.vocabulary)

        # 2. Check for vague questions (e.g. single broad word like 'employ' or 'leave')
        if is_vague_question(corrected):
            suggestion = get_vague_question_suggestion(corrected)
            return {
                "answer": suggestion,
                "original_question": user_question,
                "corrected_question": corrected,
                "retrieved_chunks": [],
                "retrieval_mode": "clarification",
                "debug_info": None
            }

        # 3. Exact Executive Role Lookup (CEO / COO) ONLY for identity/name inquiries
        exact_role_answer = extract_company_roles(corrected, self.chunks)
        if exact_role_answer:
            return {
                "answer": exact_role_answer,
                "original_question": user_question,
                "corrected_question": corrected,
                "retrieved_chunks": [],
                "retrieval_mode": "exact_lookup",
                "debug_info": None
            }

        # 4. Hybrid Retrieval (FAISS + BM25 + RRF + Cross-Encoder)
        if debug:
            retrieved_chunks, debug_info = self.retriever.retrieve(
                query=corrected,
                top_k=3,
                candidate_k=20,
                rrf_k=60,
                rerank=True,
                return_debug=True
            )
        else:
            retrieved_chunks = self.retriever.retrieve(
                query=corrected,
                top_k=3,
                candidate_k=20,
                rrf_k=60,
                rerank=True,
                return_debug=False
            )
            debug_info = None

        # 5. Generative Answer Synthesis via Ollama
        answer = self.answer_generator.generate(
            user_question=corrected,
            retrieved_chunks=retrieved_chunks
        )

        return {
            "answer": answer,
            "original_question": user_question,
            "corrected_question": corrected,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_mode": "hybrid_rag",
            "debug_info": debug_info
        }

    def start_cli(self, debug_mode: bool = False) -> None:
        """
        Start the interactive command-line interface loop.

        Args:
            debug_mode: Whether to start with debug telemetry enabled.
        """
        print("\n" + "=" * 65)
        print("  PDF RAG CHATBOT - Interactive Assistant")
        print("  Type your question below, or type 'exit' or 'quit' to stop.")
        print("  Tip: Type '/debug' to toggle retrieval telemetry on/off.")
        print("=" * 65)

        is_debug = debug_mode

        while True:
            try:
                question = input("\nUser Question: ").strip()

                if question.lower() in {"exit", "quit", "q"}:
                    print("\nExiting chatbot. Thank you!")
                    break

                if question.lower() == "/debug":
                    is_debug = not is_debug
                    state = "ENABLED" if is_debug else "DISABLED"
                    print(f"\n[Debug Mode is now {state}]")
                    continue

                if not question:
                    continue

                response = self.ask(question, debug=is_debug)

                if (
                    response["corrected_question"]
                    and response["corrected_question"] != response["original_question"]
                ):
                    print(f"[Note: Interpreted as '{response['corrected_question']}']")

                # Print debug telemetry if enabled
                if is_debug and response["debug_info"]:
                    dbg = response["debug_info"]
                    print("\n--- [DEBUG TELEMETRY] ---")
                    print(f"Original Query      : {dbg['original_query']}")
                    print(f"Expanded Query      : {dbg['expanded_query']}")
                    print(f"FAISS Candidates    : {dbg['faiss_candidates_count']}")
                    print(f"BM25 Candidates     : {dbg['bm25_candidates_count']}")
                    print(f"Hybrid Candidates   : {dbg['hybrid_candidates_count']}")
                    if dbg["cross_encoder_scores"]:
                        print("Cross-Encoder Scores:")
                        for s in dbg["cross_encoder_scores"]:
                            print(f"  Chunk {s['chunk_id']} (Page {s['page_number']}): {s['score']:.4f}")
                    print("-------------------------")

                print("\nAI Answer:")
                print(response["answer"])

                # Display sources if retrieved via RAG
                if response["retrieved_chunks"]:
                    print("\nSources Referenced:")
                    for idx, chunk in enumerate(response["retrieved_chunks"], start=1):
                        meta = chunk.get("metadata", {})
                        page = meta.get("page_number", "N/A")
                        cid = chunk.get("chunk_id", "N/A")
                        score = chunk.get("score", 0.0)
                        print(f"  [{idx}] Page {page} (Chunk {cid}, Score: {score:.4f})")

                print("-" * 65)

            except KeyboardInterrupt:
                print("\nSession interrupted. Exiting...")
                break
            except Exception as err:
                logger.error(f"Error handling question: {err}", exc_info=True)
                print(f"\nAn error occurred: {err}. Please try again.")
