"""
Dependency injection for the FastAPI application.

This module creates and manages the singleton RAG application
instance used by all API requests.
"""

from __future__ import annotations

from threading import Lock

from rag.application import RAGApplication
from rag.logger import get_logger

logger = get_logger(__name__)


_rag_application: RAGApplication | None = None
_initialization_lock = Lock()


def get_rag_application() -> RAGApplication:
    """
    Return the singleton RAG application.

    Initialization is protected by a process-level lock so that
    concurrent requests cannot initialize the RAG application
    more than once.
    """

    global _rag_application

    if _rag_application is not None:
        return _rag_application

    with _initialization_lock:
        if _rag_application is not None:
            return _rag_application

        logger.info("Initializing RAG application...")

        application = RAGApplication()

        try:
            application.load_existing()
        except Exception:
            logger.exception(
                "Failed to initialize the RAG application."
            )
            raise

        _rag_application = application

        logger.info(
            "RAG application initialized successfully."
        )

        return _rag_application


def get_application() -> RAGApplication:
    """
    FastAPI dependency.

    Return the shared, fully initialized RAG application.
    """

    return get_rag_application()
