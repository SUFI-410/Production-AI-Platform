"""
Cross Encoder reranker.

Reorders retrieved documents using a cross encoder model.
"""

from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag.config import Config
from rag.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker.

    Uses the model configured in Config.RERANKER_MODEL.
    """

    def __init__(self) -> None:

        model_name = Config.RERANKER_MODEL

        logger.info(
            "Loading reranker model: %s",
            model_name,
        )

        self.model = CrossEncoder(model_name)

        logger.info(
            "Reranker ready."
        )

    # ---------------------------------------------------------
    # Rerank
    # ---------------------------------------------------------

    def rerank(
        self,
        question: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Rerank retrieved documents.
        """

        if not documents:
            return []

        if top_k is None:
            top_k = Config.RERANK_TOP_K

        logger.info(
            "Reranking %s document(s)...",
            len(documents),
        )

        pairs = [
            (
                question,
                document.page_content,
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        logger.info("=" * 60)
        logger.info("Cross-Encoder Ranking")
        logger.info("=" * 60)

        for index, (score, document) in enumerate(
            ranked[:top_k],
            start=1,
        ):
            logger.info(
                "[%s] %.6f | %s",
                index,
                float(score),
                document.metadata.get(
                    "source",
                    "Unknown",
                ),
            )

        logger.info("=" * 60)

        results = [
            document
            for _, document in ranked[:top_k]
        ]

        logger.info(
            "Returning top %s reranked document(s).",
            len(results),
        )

        return results
