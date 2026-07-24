"""
Source formatting utilities.

Responsibilities:
- Convert LangChain Documents into API source objects.
- Remove duplicate documents.
- Extract useful metadata.
"""

from __future__ import annotations

from langchain_core.documents import Document


class SourceFormatter:
    """
    Formats retrieved documents into API response sources.
    """

    @staticmethod
    def format(
        documents: list[Document],
    ) -> list[dict]:
        """
        Return unique source metadata for API responses.
        """

        seen: set[str] = set()
        results: list[dict] = []

        for document in documents:

            metadata = document.metadata

            document_name = metadata.get(
                "file_name",
                metadata.get(
                    "source",
                    "Unknown",
                ),
            )

            if document_name in seen:
                continue

            seen.add(document_name)

            score = float(
                metadata.get(
                    "rerank_score",
                    0.0,
                )
            )

            results.append(
                {
                    "document": document_name,
                    "score": round(score, 6),
                    "metadata": {
                        "source": metadata.get(
                            "source",
                            "Unknown",
                        ),
                        "page": str(
                            metadata.get(
                                "page",
                                "-",
                            )
                        ),
                        "chunk_index": metadata.get(
                            "chunk_index",
                        ),
                        "document_type": metadata.get(
                            "document_type",
                        ),
                    },
                }
            )

        return results
