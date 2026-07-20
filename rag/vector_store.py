"""
Production Chroma Vector Store.

Responsible for:

- Creating Chroma database
- Loading existing database
- Indexing documents
- Managing collections
- Creating retrievers
"""

from __future__ import annotations

import shutil

from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag.bm25 import BM25Retriever
from rag.hybrid import HybridRetriever
from rag.config import Config
from rag.embeddings import EmbeddingFactory
from rag.exceptions import (
    CollectionNotFoundError,
    VectorStoreNotInitializedError,
)
from rag.logger import get_logger
from rag.splitter import DocumentSplitter

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Production Chroma vector database manager.
    """

    def __init__(self) -> None:
        self.embeddings = EmbeddingFactory.create()
        self.vectorstore: Chroma | None = None
        self.bm25 = BM25Retriever()

    # ---------------------------------------------------------
    # Private
    # ---------------------------------------------------------

    @staticmethod
    def database_exists() -> bool:
        """
        Check whether a Chroma database exists.
        """
        return (
            Config.CHROMA_DIR.exists()
            and any(Config.CHROMA_DIR.iterdir())
        )

    def _rebuild_bm25(self) -> None:
        """
        Rebuild the BM25 index from all chunks stored in Chroma.
        """

        if self.vectorstore is None:
            raise VectorStoreNotInitializedError(
                "Vector database has not been initialized."
            )

        data = self.vectorstore.get(
            include=["documents", "metadatas"]
        )

        documents = [
            Document(
                page_content=text,
                metadata=metadata or {},
            )
            for text, metadata in zip(
                data["documents"],
                data["metadatas"],
            )
        ]

        self.bm25.build(documents)

        logger.info(
            "Rebuilt BM25 index from %s chunk(s).",
            len(documents),
        )

    # ---------------------------------------------------------
    # Create
    # ---------------------------------------------------------

    def create(
        self,
        documents: list[Document],
    ) -> Chroma:
        """
        Create a new vector database.
        """

        logger.info("Creating vector database...")

        splitter = DocumentSplitter()

        chunks = splitter.split(documents)

        splitter.statistics(chunks)

        self.bm25.build(chunks)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=Config.CHROMA_COLLECTION,
            persist_directory=str(Config.CHROMA_DIR),
        )

        logger.info(
            "Indexed %s chunks.",
            len(chunks),
        )

        return self.vectorstore

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(self) -> Chroma:
        """
        Load an existing Chroma database.
        """

        if not self.database_exists():
            raise CollectionNotFoundError(
                "No Chroma database found."
            )

        logger.info(
            "Loading existing Chroma database..."
        )

        self.vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(Config.CHROMA_DIR),
        )

        self._rebuild_bm25()

        logger.info("Database loaded.")

        return self.vectorstore

    # ---------------------------------------------------------
    # Load or Create
    # ---------------------------------------------------------

    def load_or_create(
        self,
        documents: list[Document],
    ) -> Chroma:
        """
        Load an existing database or create a new one.
        """

        if self.database_exists():
            logger.info(
                "Existing database detected."
            )
            return self.load()

        return self.create(documents)

    # ---------------------------------------------------------
    # Retriever
    # ---------------------------------------------------------

    def as_retriever(
        self,
        metadata_filter: dict[str, str] | None = None,
    ):
        """
        Return a Hybrid Retriever
        (Chroma + BM25).
        """

        if self.vectorstore is None:
            raise VectorStoreNotInitializedError(
                "Vector database has not been initialized."
            )

        search_kwargs = {
            "k": Config.TOP_K,
            "fetch_k": Config.FETCH_K,
            "lambda_mult": Config.LAMBDA_MULT,
        }

        if metadata_filter is not None:
            search_kwargs["filter"] = metadata_filter

        chroma_retriever = (
            self.vectorstore.as_retriever(
                search_type=Config.SEARCH_TYPE,
                search_kwargs=search_kwargs,
            )
        )

        return HybridRetriever(
            chroma_retriever=chroma_retriever,
            bm25_retriever=self.bm25,
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def document_count(self) -> int:
        """
        Return the number of indexed chunks.
        """

        if self.vectorstore is None:
            raise VectorStoreNotInitializedError(
                "Vector database has not been initialized."
            )

        try:
            return self.vectorstore._collection.count()

        except AttributeError:

            collection = self.vectorstore.get()

            return len(
                collection.get("ids", [])
            )

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self) -> None:
        """
        Delete the Chroma database.
        """

        if Config.CHROMA_DIR.exists():

            shutil.rmtree(
                Config.CHROMA_DIR,
                ignore_errors=True,
            )

            Config.CHROMA_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            logger.info(
                "Vector database removed."
            )

            self.vectorstore = None

    # ---------------------------------------------------------
    # Add Documents
    # ---------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
    ) -> None:
        """
        Add new documents to an existing database.
        """

        if self.vectorstore is None:
            raise VectorStoreNotInitializedError(
                "Vector database has not been initialized."
            )

        splitter = DocumentSplitter()

        chunks = splitter.split(documents)

        self.vectorstore.add_documents(chunks)

        self._rebuild_bm25()

        self.persist()

        logger.info(
            "Added %s new chunk(s).",
            len(chunks),
        )

    # ---------------------------------------------------------
    # Persist
    # ---------------------------------------------------------

    def persist(self) -> None:
        """
        Persist the database if supported.
        """

        if self.vectorstore is None:
            return

        logger.info(
            "Database persistence handled automatically."
        )
