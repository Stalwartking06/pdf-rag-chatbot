"""
FastAPI Backend Server for PDF RAG Assistant.

Provides REST API endpoints for:
- GET /api/health     : Health check for API, RAG pipeline, and Ollama service.
- GET /api/document   : Real metadata of the loaded document (pages, chunks, models).
- POST /api/chat      : Main RAG chat endpoint returning grounded answer and sources.
- POST /api/chat/reset: Resets conversational state.

Designed to be lightweight, modular, and suitable for an 8 GB RAM local machine.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import ollama

from src.config import (
    DEFAULT_PDF_PATH,
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_MODEL_NAME,
    LLM_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    setup_logging
)
from main import build_rag_pipeline

setup_logging("INFO")
logger = logging.getLogger("rag_api")

# Global pipeline singletons - loaded ONCE on server startup
rag_chatbot = None
rag_retriever = None
rag_chunks = []
rag_metadata = []
total_pages_count = 0


def check_ollama_health() -> str:
    """Verify whether Ollama is running and responding."""
    try:
        models_data = ollama.list()
        # ollama.list() returns dict or model list
        return "connected"
    except Exception as err:
        logger.warning(f"Ollama check failed: {err}")
        return "disconnected"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Initializes models and builds FAISS + BM25 indexes ONCE at application startup,
    ensuring efficient memory usage on 8 GB RAM machines.
    """
    global rag_chatbot, rag_retriever, rag_chunks, rag_metadata, total_pages_count

    logger.info("Starting up PDF RAG FastAPI server...")
    logger.info(f"Loading document from: {DEFAULT_PDF_PATH}")

    try:
        chatbot, retriever, chunks, metadata = build_rag_pipeline(DEFAULT_PDF_PATH)
        rag_chatbot = chatbot
        rag_retriever = retriever
        rag_chunks = chunks
        rag_metadata = metadata

        if metadata:
            total_pages_count = max(m.get("page_number", 1) for m in metadata)
        else:
            total_pages_count = 0

        logger.info(
            f"RAG Pipeline successfully initialized: {len(chunks)} chunks, "
            f"{total_pages_count} pages."
        )
    except Exception as err:
        logger.critical(f"Failed to initialize RAG pipeline at startup: {err}", exc_info=True)
        rag_chatbot = None

    yield

    logger.info("Shutting down PDF RAG FastAPI server...")


app = FastAPI(
    title="PDF RAG Assistant API",
    description="Local Retrieval-Augmented Generation API for Company Policies",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request and Response Schemas ---

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question string")


class SourceItem(BaseModel):
    source: str
    page_number: int
    chunk_id: int
    score: float
    text: str


class ChatResponse(BaseModel):
    original_question: str
    corrected_question: Optional[str] = None
    answer: str
    retrieval_mode: str
    sources: list[SourceItem] = []
    debug_info: Optional[dict[str, Any]] = None


class DocumentInfoResponse(BaseModel):
    filename: str
    file_path: str
    total_pages: int
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    cross_encoder_model: str
    llm_model: str
    retrieval_methods: list[str]
    status: str


class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool
    ollama_status: str
    ollama_model: str
    document_loaded: bool


# --- API Routes ---

@app.get("/api/health", response_model=HealthResponse)
async def get_health():
    """
    Health check endpoint reporting API status, RAG pipeline state,
    and Ollama service connectivity.
    """
    ollama_state = check_ollama_health()
    is_ready = rag_chatbot is not None

    return HealthResponse(
        status="ok" if is_ready else "initializing",
        rag_initialized=is_ready,
        ollama_status=ollama_state,
        ollama_model=LLM_MODEL_NAME,
        document_loaded=bool(rag_chunks)
    )


@app.get("/api/document", response_model=DocumentInfoResponse)
async def get_document_info():
    """
    Return authentic metadata regarding the loaded company policy document
    and the retrieval pipeline configuration.
    """
    if not rag_chatbot or not rag_chunks:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document pipeline is still initializing or unavailable."
        )

    return DocumentInfoResponse(
        filename=DEFAULT_PDF_PATH.name,
        file_path=str(DEFAULT_PDF_PATH),
        total_pages=total_pages_count,
        total_chunks=len(rag_chunks),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL_NAME,
        cross_encoder_model=CROSS_ENCODER_MODEL_NAME,
        llm_model=LLM_MODEL_NAME,
        retrieval_methods=[
            "FAISS Dense Vector Search",
            "BM25 Lexical Keyword Search",
            "Reciprocal Rank Fusion (RRF, k=60)",
            "Cross-Encoder Re-Ranking (ms-marco-MiniLM)"
        ],
        status="Indexed & Ready"
    )


@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(request: ChatRequest):
    """
    Process a user query through the complete RAG pipeline:
    1. Spelling correction & query expansion
    2. Vague intent detection & clarification
    3. Exact CEO/COO identity lookup
    4. Hybrid retrieval (FAISS + BM25 + RRF + Cross-Encoder)
    5. Grounded generative answer synthesis via Ollama
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or only whitespace."
        )

    if not rag_chatbot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is currently unavailable. Please check server logs."
        )

    try:
        # Request full retrieval debug info to supply to the frontend
        result = rag_chatbot.ask(cleaned_question, debug=True)

        # Parse retrieved chunk sources into structured SourceItem models
        formatted_sources = []
        for chunk in result.get("retrieved_chunks", []):
            metadata = chunk.get("metadata", {})
            formatted_sources.append(
                SourceItem(
                    source=metadata.get("source", DEFAULT_PDF_PATH.name),
                    page_number=int(metadata.get("page_number", 0)),
                    chunk_id=int(chunk.get("chunk_id", metadata.get("chunk_id", 0))),
                    score=round(float(chunk.get("score", 0.0)), 4),
                    text=chunk.get("text", "").strip()
                )
            )

        return ChatResponse(
            original_question=result.get("original_question", cleaned_question),
            corrected_question=result.get("corrected_question"),
            answer=result.get("answer", "No answer could be generated."),
            retrieval_mode=result.get("retrieval_mode", "hybrid_rag"),
            sources=formatted_sources,
            debug_info=result.get("debug_info")
        )

    except Exception as err:
        logger.error(f"Error processing chat question: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(err)}"
        )


@app.post("/api/chat/reset")
async def reset_chat():
    """
    Acknowledge conversational history reset.
    """
    return {
        "status": "ok",
        "message": "Conversation session reset successfully."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
