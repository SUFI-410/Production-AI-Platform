from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag.adaptive_retrieval import AdaptiveRetrieval
from rag.bm25 import BM25Retriever
from rag.fusion import ReciprocalRankFusion
from rag.logger import get_logger
from rag.source_formatter import SourceFormatter

logger = get_logger(__name__)


class HybridRetriever(RunnableLambda):
    """
    Hybrid Retriever.

    Features
    --------
    - Multi Query Retrieval
    - Chroma Search
    - BM25 Search
    - Reciprocal Rank Fusion
    - Adaptive Retrieval (Dynamic Top-K)
    """

    def __init__(
        self,
        chroma_retriever,
        bm25_retriever: BM25Retriever,
    ) -> None:

        self.chroma = chroma_retriever
        self.bm25 = bm25_retriever
        self.rrf = ReciprocalRankFusion()
        self.adaptive = AdaptiveRetrieval()

        super().__init__(self.retrieve)

    # ---------------------------------------------------------
    # Retrieve
    # ---------------------------------------------------------

    def retrieve(
        self,
        queries: str | list[str],
    ) -> list[Document]:
        """
        Retrieve documents using multiple queries and
        combine the results with Reciprocal Rank Fusion.
        """

        if isinstance(queries, str):
            queries = [queries]

        # -----------------------------------------------------
        # Adaptive Top-K
        # -----------------------------------------------------

        top_k = self.adaptive.top_k(
            queries[0]
        )

        logger.info(
            "Adaptive Retrieval Top-K = %d",
            top_k,
        )

        ranked_lists: list[list[Document]] = []

        for query in queries:

            logger.info(
                "Searching: %s",
                query,
            )

            semantic_docs = self.chroma.invoke(query)
            semantic_docs = semantic_docs[:top_k]

            keyword_docs = self.bm25.search(query)
            keyword_docs = keyword_docs[:top_k]

            fused = self.rrf.fuse(
                [
                    semantic_docs,
                    keyword_docs,
                ],
                top_k=top_k,
            )

            ranked_lists.append(fused)

        documents = self.rrf.fuse(
            ranked_lists,
            top_k=top_k,
        )

        logger.info(
            "Hybrid Retriever returned %d document(s).",
            len(documents),
        )

        return documents

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
