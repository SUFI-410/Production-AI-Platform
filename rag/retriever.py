"""
Retriever utilities.

Supports:
- Vector retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
- Multi-query retrieval
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.exceptions import NoDocumentsRetrievedError
from rag.fusion import ReciprocalRankFusion
from rag.logger import get_logger
from rag.source_formatter import SourceFormatter

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
        questions: str | list[str],
    ) -> list[Document]:

        if isinstance(questions, str):
            questions = [questions]

        logger.info(
            "Running retrieval for %d querie(s).",
            len(questions),
        )

        ranked_lists: list[list[Document]] = []

        for question in questions:

            logger.info(
                "Searching: %s",
                question,
            )

            # ---------------------------------
            # Vector Search
            # ---------------------------------

            vector_docs = self.vector_retriever.invoke(
                question
            )

            # ---------------------------------
            # Hybrid Search
            # ---------------------------------

            if self.hybrid_search is not None:

                bm25_docs = self.hybrid_search.search(
                    question
                )

                ranked_lists.append(vector_docs)
                ranked_lists.append(bm25_docs)

            else:

                ranked_lists.append(vector_docs)

        documents = self.rrf.fuse(ranked_lists)

        if not documents:

            raise NoDocumentsRetrievedError(
                "No relevant documents found."
            )

        logger.info(
            "Retrieved %d fused document(s).",
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
                "[%d]\n%s\n",
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
        """
        Return formatted source information.
        """

        return SourceFormatter.format(
            documents
        )
