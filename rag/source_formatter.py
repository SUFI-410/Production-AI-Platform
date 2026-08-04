"""
Source formatting utilities.

Responsibilities:
- Convert LangChain Documents into API source objects.
- Remove duplicate documents.
- Extract useful metadata.
- Provide consistent source identities for citations.
"""

from __future__ import annotations

from langchain_core.documents import Document


class SourceFormatter:
    """
    Formats retrieved documents into API response sources.
    """

    @staticmethod
    def document_name(
        document: Document,
    ) -> str:
        """
        Return the canonical identity used for source numbering.

        Context citations and API sources must both use this
        value so their numbering remains aligned.
        """

        metadata = document.metadata

        document_name = (
            metadata.get("file_name")
            or metadata.get("source")
            or "Unknown"
        )

        return str(document_name)

    @classmethod
    def format(
        cls,
        documents: list[Document],
    ) -> list[dict]:
        """
        Return unique source metadata in first-appearance order.

        Reranked documents are already ordered by relevance, so
        the first occurrence also supplies the source's best score.
        """

        seen: set[str] = set()
        results: list[dict] = []

        for document in documents:
            metadata = document.metadata
            document_name = cls.document_name(document)

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
