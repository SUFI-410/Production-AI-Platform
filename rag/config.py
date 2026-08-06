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

    # Default retrieval size.
    # Adaptive Retrieval may override this value.
    TOP_K = 8

    # Number of candidate documents fetched before MMR.
    FETCH_K = 20

    # Diversity parameter for MMR.
    # 0.0 = Maximum diversity
    # 1.0 = Maximum relevance
    LAMBDA_MULT = 0.5

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------

    RERANKER_MODEL = "BAAI/bge-reranker-base"

    # Maximum documents kept after reranking.
    RERANK_TOP_K = 5

    # Minimum CrossEncoder relevance score required
    # for a document to be considered relevant.
    RERANK_THRESHOLD = 0.05

    # ------------------------------------------------------------------
    # Response Cache
    # ------------------------------------------------------------------

    # Cache lifetime (seconds)
    CACHE_TTL = 3600

    # ------------------------------------------------------------------
    # Conversation Sessions
    # ------------------------------------------------------------------

    # Inactive session lifetime in seconds.
    SESSION_TTL = int(
        os.getenv(
            "SESSION_TTL",
            "3600",
        )
    )

    # Maximum number of conversation sessions kept in memory.
    SESSION_MAX_SESSIONS = int(
        os.getenv(
            "SESSION_MAX_SESSIONS",
            "1000",
        )
    )

    # Maximum number of user/assistant messages per session.
    SESSION_MAX_MESSAGES = int(
        os.getenv(
            "SESSION_MAX_MESSAGES",
            "10",
        )
    )

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
    # CORS
    # ------------------------------------------------------------------

    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173",
        ).split(",")
    ]

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
