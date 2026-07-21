"""
Reciprocal Rank Fusion (RRF).

Combines multiple ranked retrieval results into a
single ranked list.

Reference:
Cormack, Clarke & Buettcher (2009)
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document

from rag.logger import get_logger

logger = get_logger(__name__)


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion implementation.

    Formula:

        score += 1 / (k + rank)

    where k is typically 60.
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    # ---------------------------------------------------------
    # Score
    # ---------------------------------------------------------

    def score(
        self,
        rank: int,
    ) -> float:
        """
        Calculate RRF score for one ranked position.
        """

        return 1.0 / (self.k + rank)

    # ---------------------------------------------------------
    # Deduplicate
    # ---------------------------------------------------------

    @staticmethod
    def deduplicate(
        documents: list[Document],
    ) -> list[Document]:
        """
        Remove duplicate documents.
        """

        seen: set[str] = set()
        unique: list[Document] = []

        for document in documents:

            key = (
                document.page_content,
                tuple(
                    sorted(
                        (
                            (k, str(v))
                            for k, v in document.metadata.items()
                        ),
                        key=lambda item: item[0],
                    )
                ),
            )

            if str(key) in seen:
                continue

            seen.add(str(key))
            unique.append(document)

        return unique

    # ---------------------------------------------------------
    # Fuse
    # ---------------------------------------------------------

    def fuse(
        self,
        ranked_lists: list[list[Document]],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            ranked_lists:
                Multiple ranked retrieval results.

            top_k:
                Number of documents to return.
        """

        logger.info(
            "Running Reciprocal Rank Fusion..."
        )

        scores: defaultdict[str, float] = defaultdict(float)
        lookup: dict[str, Document] = {}

        for ranked in ranked_lists:

            for rank, document in enumerate(
                ranked,
                start=1,
            ):

                key = str(
                    (
                        document.page_content,
                        tuple(
                            sorted(
                                (
                                    (k, str(v))
                                    for k, v in document.metadata.items()
                                ),
                                key=lambda item: item[0],
                            )
                        ),
                    )
                )

                scores[key] += self.score(rank)
                lookup[key] = document

        ranked_keys = sorted(
            scores.keys(),
            key=lambda key: scores[key],
            reverse=True,
        )

        results = [
            lookup[key]
            for key in ranked_keys
        ]

        results = self.deduplicate(results)

        logger.info(
            "RRF returned %s document(s).",
            min(top_k, len(results)),
        )

        return results[:top_k]
