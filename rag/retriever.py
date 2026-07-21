"""
Retriever utilities.

Supports:
- Vector retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.exceptions import NoDocumentsRetrievedError
from rag.fusion import ReciprocalRankFusion
from rag.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Production retriever wrapper.
    """

    def __init__(
        self,
        vector_retriever,
        hybrid_search=None,
    ):
        self.vector_retriever = vector_retriever
        self.hybrid_search = hybrid_search
        self.rrf = ReciprocalRankFusion()

    # ---------------------------------------------------------
    # Retrieve
    # ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
    ) -> list[Document]:

        logger.info(
            "Searching: %s",
            question,
        )

        # -----------------------------
        # Vector Search
        # -----------------------------

        vector_docs = self.vector_retriever.invoke(
            question
        )

        # -----------------------------
        # Hybrid Search
        # -----------------------------

        if self.hybrid_search is not None:

            bm25_docs = self.hybrid_search.search(
                question
            )

            documents = self.rrf.fuse(
                [vector_docs, bm25_docs]
            )

        else:

            documents = vector_docs

        if not documents:

            raise NoDocumentsRetrievedError(
                "No relevant documents found."
            )

        logger.info(
            "Retrieved %s document(s).",
            len(documents),
        )

        return documents

    # ---------------------------------------------------------
    # Preview
    # ---------------------------------------------------------

    @staticmethod
    def preview(
        documents: list[Document],
        limit: int = 3,
    ) -> None:

        logger.info("=" * 60)
        logger.info("Retrieved Context")
        logger.info("=" * 60)

        for i, document in enumerate(
            documents[:limit],
            start=1,
        ):

            logger.info(
                "[%s]\n%s\n",
                i,
                document.page_content[:300],
            )

        logger.info("=" * 60)

    # ---------------------------------------------------------
    # Sources
    # ---------------------------------------------------------

    @staticmethod
    def sources(
        documents: list[Document],
    ) -> list[dict]:

        seen = set()
        results = []

        for document in documents:

            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page",
                "-",
            )

            key = (
                source,
                page,
            )

            if key in seen:
                continue

            seen.add(key)

            results.append(
                {
                    "source": source,
                    "page": page,
                }
            )

        return results
