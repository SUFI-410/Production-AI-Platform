"""
Production RAG Chain using LCEL.

Responsibilities:
- Retrieve relevant documents
- Build context
- Invoke GPT model
- Stream GPT responses
- Return answer and sources
"""

from __future__ import annotations

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

from rag.config import Config
from rag.logger import get_logger
from rag.prompt import PromptFactory
from rag.utils import format_documents

logger = get_logger(__name__)


class RAGChain:
    """
    Production Retrieval-Augmented Generation chain.
    """

    def __init__(self, retriever, reranker):

        self.retriever = retriever

        self.reranker = reranker

        self.llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE,
        )

        self.prompt = PromptFactory.create()

        self.chain = self._build_chain()

    # ---------------------------------------------------------
    # Context Preparation
    # ---------------------------------------------------------

    def _prepare_context(
        self,
        inputs: dict,
    ) -> str:
        """
        Prepare the context for the LLM.

        Steps:
        1. Receive retrieved documents.
        2. Re-rank them using the Cross Encoder.
        3. Keep the top documents.
        4. Format them into the prompt context.
        """

        question: str = inputs["question"]
        documents: list[Document] = inputs["documents"]

        documents = self.reranker.rerank(
            question=question,
            documents=documents,
        )

        return format_documents(documents)

    # ---------------------------------------------------------
    # Build LCEL Chain
    # ---------------------------------------------------------

    def _build_chain(self):

        context_chain = (
            RunnableParallel(
                question=itemgetter("question"),
                documents=itemgetter("question") | self.retriever,
            )
            | RunnableLambda(self._prepare_context)
        )

        return (
            {
                "context": context_chain,
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # ---------------------------------------------------------
    # Ask
    # ---------------------------------------------------------

    def invoke(
        self,
        question: str,
    ) -> str:
        """
        Generate a complete answer.
        """

        logger.info("Question: %s", question)

        answer = self.chain.invoke(
            {
                "question": question,
            }
        )

        logger.info("Answer generated.")

        return answer

    # ---------------------------------------------------------
    # Stream
    # ---------------------------------------------------------

    def stream(
        self,
        question: str,
    ):
        """
        Stream the generated answer chunk by chunk.
        """

        logger.info("Streaming question: %s", question)

        yield from self.chain.stream(
            {
                "question": question,
            }
        )

    # ---------------------------------------------------------
    # Retrieve Only
    # ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
    ) -> list[Document]:
        """
        Return retrieved documents only.
        """

        return self.retriever.invoke(question)

    # ---------------------------------------------------------
    # Answer + Sources
    # ---------------------------------------------------------

    def ask(
        self,
        question: str,
    ) -> dict:
        """
        Return both answer and retrieved documents.
        """

        documents = self.retrieve(question)

        answer = self.invoke(question)

        return {
            "question": question,
            "answer": answer,
            "documents": documents,
        }
