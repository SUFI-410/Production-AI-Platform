"""
Application configuration.

All configurable values live here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    CHAT_MODEL = "gpt-5-mini"

    EMBEDDING_MODEL = "text-embedding-3-small"

    TEMPERATURE = 0.0

    MAX_TOKENS = 1024

    # ------------------------------------------------------------------
    # Chroma
    # ------------------------------------------------------------------

    CHROMA_DIR = Path("chroma_db")

    CHROMA_COLLECTION = "documents"

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    CHUNK_SIZE = 1000

    CHUNK_OVERLAP = 200

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    SEARCH_TYPE = "mmr"

    TOP_K = 8

    FETCH_K = 20

    LAMBDA_MULT = 0.5

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------

    RERANKER_MODEL = "BAAI/bge-reranker-base"

    RERANK_TOP_K = 5

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    LOG_DIR = Path("logs")

    LOG_FILE = LOG_DIR / "rag.log"

    LOG_LEVEL = "INFO"

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    USER_AGENT = (
        "Production-RAG/1.0 "
        "(https://github.com/yourusername/production-rag)"
    )

    REQUEST_TIMEOUT = 30

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""

        if not cls.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY was not found.\n"
                "Create a .env file containing:\n\n"
                "OPENAI_API_KEY=your_api_key"
            )

        cls.CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        cls.LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
