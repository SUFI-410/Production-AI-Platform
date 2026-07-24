"""
Dependency injection for the FastAPI application.

This module creates and manages the singleton RAG application
instance used by all API requests.
"""

from __future__ import annotations

from functools import lru_cache

from rag.application import RAGApplication
from rag.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_rag_application() -> RAGApplication:
    """
    Return the singleton RAG application.

    The application is created only once during the
    process lifetime and reused for every request.
    """

    logger.info("Initializing RAG application...")

    app = RAGApplication()

    try:
        app.load_existing()
        logger.info("Existing vector database loaded.")
    except Exception as exc:
        logger.warning(
            "No existing vector database found: %s",
            exc,
        )

    return app


def get_application() -> RAGApplication:
    """
    FastAPI dependency.

    Returns the shared RAG application instance.
    """

    return get_rag_application()
