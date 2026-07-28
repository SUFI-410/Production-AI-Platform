"""
Main RAG application orchestrator.
"""

from __future__ import annotations

from pathlib import Path

from rag.chain import RAGChain
from rag.config import Config
from rag.loader import DocumentLoader
from rag.logger import get_logger
from rag.memory import ConversationMemory
from rag.multi_query import MultiQueryGenerator
from rag.query_rewriter import QueryRewriter
from rag.reranker import Reranker
from rag.response_cache import ResponseCache
from rag.vector_store import VectorStoreManager

logger = get_logger(__name__)


class RAGApplication:
    """
    High-level interface for the Production RAG system.
    """

    def __init__(self) -> None:

        self.vector_manager = VectorStoreManager()

        self.reranker = Reranker()

        self.memory = ConversationMemory()

        self.query_rewriter = QueryRewriter()

        self.multi_query = MultiQueryGenerator()

        self.cache = ResponseCache(
            ttl_seconds=Config.CACHE_TTL,
        )

        self.chain: RAGChain | None = None

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _create_chain(
        self,
        metadata_filter: dict[str, str] | None = None,
    ) -> RAGChain:
        """
        Create a RAG chain using the requested metadata filter.
        """

        retriever = self.vector_manager.as_retriever(
            metadata_filter=metadata_filter,
        )

        return RAGChain(
            retriever=retriever,
            reranker=self.reranker,
            memory=self.memory,
        )

    def _prepare_question(
        self,
        question: str,
    ) -> str:
        """
        Rewrite the question using conversation history
        when history is available.
        """

        history = self.memory.formatted_history()

        if not history.strip():
            return question

        return self.query_rewriter.rewrite(
            question=question,
            history=history,
        )

    def _cache_key(
        self,
        question: str,
        metadata_filter: dict[str, str] | None,
    ) -> str:
        """
        Build a cache key that also includes metadata filters.
        """

        if metadata_filter is None:
            return question

        return f"{question}|{sorted(metadata_filter.items())}"

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def initialize_pdf(
        self,
        pdf_path: str | Path,
    ) -> None:

        logger.info("Initializing from PDF...")

        documents = DocumentLoader.load_pdf(pdf_path)

        self.vector_manager.load_or_create(documents)

        self.chain = self._create_chain()

        logger.info("Application initialized.")

    def initialize_pdfs(
        self,
        pdfs: list[str | Path],
    ) -> None:

        logger.info("Initializing from PDFs...")

        documents = DocumentLoader.load_pdfs(pdfs)

        self.vector_manager.load_or_create(documents)

        self.chain = self._create_chain()

        logger.info("Application initialized.")

    def initialize_web(
        self,
        url: str,
    ) -> None:

        logger.info("Initializing from website...")

        documents = DocumentLoader.load_web(url)

        self.vector_manager.load_or_create(documents)

        self.chain = self._create_chain()

        logger.info("Application initialized.")

    def initialize_sources(
        self,
        sources: list[dict],
    ) -> None:

        logger.info("Initializing from mixed sources...")

        documents = DocumentLoader.load_sources(sources)

        self.vector_manager.load_or_create(documents)

        self.chain = self._create_chain()

        logger.info("Application initialized.")

    def load_existing(self) -> None:

        logger.info("Loading existing database...")

        self.vector_manager.load()

        self.chain = self._create_chain()

        logger.info("Existing database loaded.")


    # ---------------------------------------------------------
    # Incremental Ingestion
    # ---------------------------------------------------------

    def add_pdf(
        self,
        pdf_path: str | Path,
    ) -> None:
        """
        Add a PDF to the existing database.
        """

        logger.info("Adding PDF: %s", pdf_path)

        documents = DocumentLoader.load_pdf(pdf_path)

        self.vector_manager.add_documents(documents)

        self.chain = self._create_chain()

        self.cache.clear()

        logger.info("PDF added successfully.")

    def add_pdfs(
        self,
        pdfs: list[str | Path],
    ) -> None:
        """
        Add multiple PDFs to the existing database.
        """

        logger.info("Adding multiple PDFs...")

        documents = DocumentLoader.load_pdfs(pdfs)

        self.vector_manager.add_documents(documents)

        self.chain = self._create_chain()

        self.cache.clear()

        logger.info("PDFs added successfully.")

    def add_web(
        self,
        url: str,
    ) -> None:
        """
        Add a website to the existing database.
        """

        logger.info("Adding website: %s", url)

        documents = DocumentLoader.load_web(url)

        self.vector_manager.add_documents(documents)

        self.chain = self._create_chain()

        self.cache.clear()

        logger.info("Website added successfully.")

    # ---------------------------------------------------------
    # Ask
    # ---------------------------------------------------------

    def ask(
        self,
        question: str,
        metadata_filter: dict[str, str] | None = None,
    ) -> str:
        """
        Return only the generated answer.
        """

        if self.chain is None:
            raise RuntimeError(
                "Application has not been initialized."
            )

        # Check cache BEFORE rewriting
        cache_key = self._cache_key(question, metadata_filter)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Response cache hit.")
            return cached["answer"]

        # Cache miss – rewrite and generate
        rewritten_question = self._prepare_question(question)
        queries = self.multi_query.generate(rewritten_question)

        chain = (
            self.chain
            if metadata_filter is None
            else self._create_chain(metadata_filter)
        )

        documents = chain.retrieve(queries)
        answer = chain.invoke(
            question=rewritten_question,
            documents=documents,
        )

        self.cache.set(
            cache_key,
            {
                "answer": answer,
                "documents": documents,
                "sources": chain.retriever.sources(
                    documents
                ),
                "grounded": True,
                "cached": False,
            },
        )

        return answer

    def ask_with_sources(
        self,
        question: str,
        metadata_filter: dict[str, str] | None = None,
    ) -> dict:
        """
        Return answer plus sources.
        """

        if self.chain is None:
            raise RuntimeError(
                "Application has not been initialized."
            )

        logger.info("=" * 70)
        logger.info("Original Question : %s", question)

        # ---- Build cache key from original question (NO rewrite yet) ----
        cache_key = self._cache_key(question, metadata_filter)

        logger.info("Cache Key : %r", cache_key)
        logger.info(
            "Cache Size: %d",
            self.cache.size(),
        )

        # ---- CHECK CACHE FIRST ----
        cached = self.cache.get(cache_key)
        if cached is not None:

            logger.info("******** CACHE HIT ********")
            cached = cached.copy()
            cached["cached"] = True
            logger.info("=" * 70)
            return cached

        logger.info("******** CACHE MISS ********")

        # ---- Only now rewrite (because we need a new answer) ----
        rewritten_question = self._prepare_question(question)
        logger.info(
            "Rewritten Question: %s",
            rewritten_question,
        )

        queries = self.multi_query.generate(
            rewritten_question
        )

        chain = (
            self.chain
            if metadata_filter is None
            else self._create_chain(metadata_filter)
        )

        documents = chain.retrieve(
            queries
        )

        result = chain.ask(
            question=rewritten_question,
            documents=documents,
        )

        result["cached"] = False

        # Store using the original question as key
        self.cache.set(
            cache_key,
            result,
        )

        logger.info("Saved response to cache.")
        logger.info(
            "Cache Size After Save: %d",
            self.cache.size(),
        )

        logger.info("=" * 70)

        return result

    # ---------------------------------------------------------
    # Database
    # ---------------------------------------------------------

    def reset_database(self) -> None:
        """
        Delete the Chroma database.
        """

        self.vector_manager.reset()

        self.cache.clear()

    def database_size(self) -> int:
        """
        Return indexed chunk count.
        """

        return self.vector_manager.document_count()
