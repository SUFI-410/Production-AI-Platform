"""
Production RAG Chain using LCEL.

Responsibilities:
- Build context
- Compress context
- Rerank documents
- Invoke GPT model
- Return answers and sources

Conversation memory is managed by RAGApplication.
"""

from __future__ import annotations

from collections.abc import Iterator
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
)
from langchain_openai import ChatOpenAI

from rag.config import Config
from rag.context_compressor import ContextCompressor
from rag.logger import get_logger
from rag.prompt import PromptFactory
from rag.utils import format_documents

logger = get_logger(__name__)

NO_RELEVANT_INFORMATION_ANSWER = (
    "I couldn't find any relevant information "
    "in the knowledge base to answer your question."
)

INSUFFICIENT_CONTEXT_ANSWER = (
    "I don't have enough information in the "
    "provided documents to answer that."
)

REFUSAL_ANSWER_MARKERS = (
    "i couldn't find any relevant information "
    "in the knowledge base",
    "i don't have enough information in the "
    "provided documents",
)


class RAGChain:
    """
    Production Retrieval-Augmented Generation chain.

    The chain is stateless with respect to conversation memory.
    Callers provide the formatted history for each request.
    """

    def __init__(
        self,
        retriever,
        reranker,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = ContextCompressor()

        self.llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE,
        )

        self.prompt = PromptFactory.create()
        self.chain = self._build_chain()

    # ---------------------------------------------------------
    # Document Preparation
    # ---------------------------------------------------------

    def _prepare_documents(
        self,
        question: str,
        documents: list[Document],
    ) -> list[Document]:
        """
        Compress and rerank retrieved documents.
        """

        documents = self.compressor.compress(
            question=question,
            documents=documents,
        )

        documents = self.reranker.rerank(
            question=question,
            documents=documents,
        )

        return documents

    # ---------------------------------------------------------
    # Context Formatting
    # ---------------------------------------------------------

    def _prepare_context(
        self,
        inputs: dict,
    ) -> str:
        """
        Convert reranked documents into prompt context.
        """

        return format_documents(
            inputs["documents"]
        )

    # ---------------------------------------------------------
    # Build LCEL Chain
    # ---------------------------------------------------------

    def _build_chain(self):
        context_chain = (
            RunnableParallel(
                documents=itemgetter("documents"),
            )
            | RunnableLambda(self._prepare_context)
        )

        return (
            {
                "history": itemgetter("history"),
                "context": context_chain,
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # ---------------------------------------------------------
    # Refusal Detection
    # ---------------------------------------------------------

    @staticmethod
    def _is_refusal_answer(
        answer: str,
    ) -> bool:
        """
        Return True for known knowledge-base refusal answers.
        """

        normalized_answer = " ".join(
            answer.casefold().split()
        )

        return any(
            marker in normalized_answer
            for marker in REFUSAL_ANSWER_MARKERS
        )

    # ---------------------------------------------------------
    # Core Generation
    # ---------------------------------------------------------

    def _generate(
        self,
        question: str,
        documents: list[Document],
        history: str = "",
    ) -> tuple[str, list[Document]]:
        """
        Generate an answer using caller-provided history.
        """

        logger.info(
            "Question: %s",
            question,
        )

        documents = self._prepare_documents(
            question,
            documents,
        )

        if not documents:
            logger.warning(
                "No document passed reranker threshold."
            )

            return NO_RELEVANT_INFORMATION_ANSWER, []

        answer = self.chain.invoke(
            {
                "question": question,
                "documents": documents,
                "history": history,
            }
        )

        logger.info(
            "Answer generated."
        )

        return answer, documents

    # ---------------------------------------------------------
    # Invoke
    # ---------------------------------------------------------

    def invoke(
        self,
        question: str,
        documents: list[Document],
        history: str = "",
    ) -> str:
        """
        Generate an answer only.
        """

        answer, _ = self._generate(
            question=question,
            documents=documents,
            history=history,
        )

        return answer

    # ---------------------------------------------------------
    # Stream
    # ---------------------------------------------------------

    def stream(
        self,
        question: str,
        documents: list[Document],
        history: str = "",
    ) -> Iterator[str]:
        """
        Stream an answer using caller-provided history.
        """

        logger.info(
            "Streaming question: %s",
            question,
        )

        documents = self._prepare_documents(
            question,
            documents,
        )

        if not documents:
            yield NO_RELEVANT_INFORMATION_ANSWER
            return

        yield from self.chain.stream(
            {
                "question": question,
                "documents": documents,
                "history": history,
            }
        )

    # ---------------------------------------------------------
    # Retrieve Only
    # ---------------------------------------------------------

    def retrieve(
        self,
        questions: str | list[str],
    ) -> list[Document]:
        """
        Return retrieved documents only.
        """

        return self.retriever.invoke(
            questions
        )

    # ---------------------------------------------------------
    # Answer + Sources
    # ---------------------------------------------------------

    def ask(
        self,
        question: str,
        documents: list[Document],
        history: str = "",
    ) -> dict:
        """
        Return answer, documents, sources, and groundedness.
        """

        answer, documents = self._generate(
            question=question,
            documents=documents,
            history=history,
        )

        if self._is_refusal_answer(answer):
            logger.info(
                "Knowledge-base refusal detected; "
                "supporting sources omitted."
            )

            documents = []

        return {
            "question": question,
            "answer": answer,
            "documents": documents,
            "sources": self.retriever.sources(
                documents
            ),
            "grounded": bool(documents),
        }
