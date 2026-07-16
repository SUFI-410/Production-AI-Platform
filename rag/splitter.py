"""
Document splitter.

Responsible for splitting documents into chunks
while preserving metadata.
"""

from __future__ import annotations

from statistics import mean

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Config
from rag.logger import get_logger

logger = get_logger(__name__)


class DocumentSplitter:
    """
    Production document splitter.
    """

    def __init__(self) -> None:

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
            keep_separator=True,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                ";",
                ",",
                " ",
                "",
            ],
        )

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    def split(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Split documents while enriching metadata.
        """

        logger.info(
            "Splitting %s document(s)...",
            len(documents),
        )

        chunks = self.splitter.split_documents(
            documents
        )

        document_chunk_counter: dict[str, int] = {}

        for chunk in chunks:

            metadata = chunk.metadata

            source = metadata.get(
                "source",
                "unknown",
            )

            document_chunk_counter.setdefault(
                source,
                0,
            )

            document_chunk_counter[source] += 1

            chunk.metadata = {

                **metadata,

                "document_id": source,

                "chunk_index":
                    document_chunk_counter[source],

                "chunk_id":
                    f"{source}::"
                    f"{document_chunk_counter[source]}",

                "chunk_size":
                    len(chunk.page_content),

                "domain":
                    metadata.get(
                        "url",
                        source,
                    ).split("/")[2]
                    if "://" in metadata.get(
                        "url",
                        source,
                    )
                    else "local",

                "knowledge_source":
                    metadata.get(
                        "document_type",
                        "unknown",
                    ),
            }

        logger.info(
            "Generated %s chunk(s).",
            len(chunks),
        )

        return chunks

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    @staticmethod
    def statistics(
        chunks: list[Document],
    ) -> None:
        """
        Log chunk statistics.
        """

        if not chunks:
            logger.warning(
                "No chunks generated."
            )
            return

        lengths = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        logger.info("=" * 60)
        logger.info("Chunk Statistics")
        logger.info("=" * 60)
        logger.info(
            "Chunks           : %s",
            len(chunks),
        )
        logger.info(
            "Average Length   : %.2f",
            mean(lengths),
        )
        logger.info(
            "Smallest Chunk   : %s",
            min(lengths),
        )
        logger.info(
            "Largest Chunk    : %s",
            max(lengths),
        )
        logger.info("=" * 60)

    # ---------------------------------------------------------
    # Preview
    # ---------------------------------------------------------

    @staticmethod
    def preview(
        chunks: list[Document],
        limit: int = 3,
    ) -> None:
        """
        Preview the first few chunks.
        """

        logger.info(
            "Previewing first %s chunk(s).",
            limit,
        )

        for index, chunk in enumerate(
            chunks[:limit],
            start=1,
        ):
            logger.info(
                "[Chunk %s]\n%s\n",
                index,
                chunk.page_content[:250],
            )
