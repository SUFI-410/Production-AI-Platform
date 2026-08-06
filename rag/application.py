"""
Main RAG application orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path

from rag.chain import RAGChain
from rag.config import Config
from rag.loader import DocumentLoader
from rag.logger import get_logger
from rag.memory import SessionMemoryStore
from rag.multi_query import MultiQueryGenerator
from rag.query_rewriter import QueryRewriter
from rag.reranker import Reranker
from rag.response_cache import ResponseCache
from rag.vector_store import VectorStoreManager

logger = get_logger(__name__)


class RAGApplication:
    """
    High-level interface for the Production RAG system.

    Conversation history is isolated by session ID. Requests
    for the same session are serialized by SessionMemoryStore.
    """

    def __init__(self) -> None:
        self.vector_manager = VectorStoreManager()
        self.reranker = Reranker()

        self.memory_store = SessionMemoryStore(
            ttl_seconds=Config.SESSION_TTL,
            max_sessions=Config.SESSION_MAX_SESSIONS,
            max_messages=Config.SESSION_MAX_MESSAGES,
        )

        # CLI and evaluation callers may omit a session ID.
        # Such calls share one private session for this application
        # instance rather than a process-wide fixed session name.
        self._default_session_id = (
            f"local-{uuid.uuid4().hex}"
        )

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
        )

    def _resolve_session_id(
        self,
        session_id: str | None,
    ) -> str:
        """
        Return a valid session ID for the request.
        """

        if session_id is None:
            return self._default_session_id

        normalized_session_id = session_id.strip()

        if not normalized_session_id:
            raise ValueError(
                "session_id must not be empty."
            )

        return normalized_session_id

    def _prepare_question(
        self,
        question: str,
        history: str,
    ) -> str:
        """
        Rewrite the question when session history is available.
        """

        if not history.strip():
            return question

        return self.query_rewriter.rewrite(
            question=question,
            history=history,
        )

    def _requires_conversation_context(
        self,
        question: str,
    ) -> bool:
        """
        Return True when a question likely refers to earlier chat.

        The detection is intentionally conservative. Marking a
        standalone question as contextual only reduces cache reuse,
        while treating a contextual question as standalone could
        return an answer from the wrong conversation point.
        """

        normalized_question = " ".join(
            question.casefold().split()
        )

        tokens = set(
            re.findall(
                r"[a-z0-9']+",
                normalized_question,
            )
        )

        contextual_tokens = {
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "they",
            "them",
            "their",
            "theirs",
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
            "former",
            "latter",
        }

        if tokens & contextual_tokens:
            return True

        contextual_phrases = (
            "what about",
            "how about",
            "follow-up",
            "follow up",
            "you said",
            "you mentioned",
            "you explained",
            "just explained",
            "the above",
            "the previous",
            "the earlier",
            "the concept",
            "this concept",
            "that concept",
            "same thing",
            "same example",
            "another example",
            "tell me more",
            "explain again",
            "first one",
            "second one",
            "last one",
        )

        return any(
            phrase in normalized_question
            for phrase in contextual_phrases
        )

    def _cache_key(
        self,
        question: str,
        metadata_filter: dict[str, str] | None,
        session_id: str,
        history: str,
    ) -> str:
        """
        Build a deterministic session-aware cache key.

        History is included because the same question can have
        a different meaning at different conversation points.
        """

        payload = {
            "session_id": session_id,
            "question": " ".join(
                question.casefold().split()
            ),
            "metadata_filter": metadata_filter or {},
            "history": history,
        }

        serialized_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized_payload.encode("utf-8")
        ).hexdigest()

    def _select_chain(
        self,
        metadata_filter: dict[str, str] | None,
    ) -> RAGChain:
        """
        Return the default or filtered chain.
        """

        if self.chain is None:
            raise RuntimeError(
                "Application has not been initialized."
            )

        if metadata_filter is None:
            return self.chain

        return self._create_chain(
            metadata_filter=metadata_filter,
        )

    def _generate_response(
        self,
        question: str,
        history: str,
        metadata_filter: dict[str, str] | None,
    ) -> dict:
        """
        Retrieve documents and generate a response.
        """

        rewritten_question = self._prepare_question(
            question=question,
            history=history,
        )

        logger.info(
            "Rewritten Question: %s",
            rewritten_question,
        )

        queries = self.multi_query.generate(
            rewritten_question
        )

        chain = self._select_chain(
            metadata_filter=metadata_filter,
        )

        documents = chain.retrieve(
            queries
        )

        return chain.ask(
            question=rewritten_question,
            documents=documents,
            history=history,
        )

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
        session_id: str | None = None,
        use_cache: bool = True,
    ) -> str:
        """
        Return only the generated answer.
        """

        result = self.ask_with_sources(
            question=question,
            metadata_filter=metadata_filter,
            session_id=session_id,
            use_cache=use_cache,
        )

        return result["answer"]

    def ask_with_sources(
        self,
        question: str,
        metadata_filter: dict[str, str] | None = None,
        session_id: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        """
        Return an answer plus sources.

        The session lock covers history reading, query rewriting,
        generation, cache handling, and memory mutation. This
        guarantees ordered exchanges within one session.
        """

        if self.chain is None:
            raise RuntimeError(
                "Application has not been initialized."
            )

        resolved_session_id = self._resolve_session_id(
            session_id
        )

        logger.info("=" * 70)
        logger.info("Original Question : %s", question)
        logger.info(
            "Session ID        : %s",
            resolved_session_id,
        )

        with self.memory_store.session(
            resolved_session_id
        ) as memory:
            history = memory.formatted_history()

            requires_context = (
                self._requires_conversation_context(
                    question
                )
            )

            effective_history = (
                history
                if requires_context
                else ""
            )

            if history and not requires_context:
                logger.info(
                    "Standalone question detected; "
                    "conversation history ignored."
                )

            cache_key = self._cache_key(
                question=question,
                metadata_filter=metadata_filter,
                session_id=resolved_session_id,
                history=effective_history,
            )

            logger.info("Cache Key : %r", cache_key)
            logger.info(
                "Cache Size: %d",
                self.cache.size(),
            )

            if use_cache:
                cached = self.cache.get(cache_key)

                if cached is not None:
                    logger.info("******** CACHE HIT ********")

                    result = cached.copy()
                    result["cached"] = True
                    result["session_id"] = resolved_session_id

                    memory.add_exchange(
                        user_message=question,
                        assistant_message=result["answer"],
                    )

                    logger.info("=" * 70)

                    return result
            else:
                logger.info("Response cache disabled.")

            logger.info("******** CACHE MISS ********")

            result = self._generate_response(
                question=question,
                history=effective_history,
                metadata_filter=metadata_filter,
            )

            result["cached"] = False
            result["session_id"] = resolved_session_id

            memory.add_exchange(
                user_message=question,
                assistant_message=result["answer"],
            )

            if use_cache:
                self.cache.set(
                    cache_key,
                    result.copy(),
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
        Delete the Chroma database and runtime state.
        """

        self.vector_manager.reset()

        self.cache.clear()
        self.memory_store.clear()

    def database_size(self) -> int:
        """
        Return indexed chunk count.
        """

        return self.vector_manager.document_count()
