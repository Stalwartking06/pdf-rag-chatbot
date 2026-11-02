"""
Configuration module for the PDF RAG Chatbot.

This module stores all paths, model names, and hyper-parameters in one place,
making them easy to modify without changing application code.
"""

from pathlib import Path
import logging
import sys

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default PDF file path
DEFAULT_PDF_PATH = DOCUMENTS_DIR / "company_policy.pdf"

# Text Chunking Settings
CHUNK_SIZE = 1000       # Maximum characters per chunk
CHUNK_OVERLAP = 200     # Overlapping characters between consecutive chunks

# Model Names
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "qwen2.5:1.5b"
LLM_TEMPERATURE = 0.0

# Retrieval & Reranking Settings
RETRIEVAL_TOP_K = 20    # Number of candidates retrieved from FAISS & BM25 each
FINAL_TOP_K = 3         # Number of final chunks passed to the LLM
RRF_K = 60              # Reciprocal Rank Fusion smoothing constant
# Rerank threshold is optional; None means do not discard candidates based on arbitrary logit cutoff
RERANK_THRESHOLD = None

# Evaluation Settings
EVAL_TOP_K = 5


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Configure application-wide logging with both file and console handlers.
    Logs are saved to logs/app.log and printed cleanly to stderr/stdout.
    """
    log_file = LOGS_DIR / "app.log"

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers if already configured
    if not root_logger.handlers:
        # File handler (detailed)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

        # Console handler (clean output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)  # Console shows warnings/errors, INFO goes to file
        root_logger.addHandler(console_handler)
