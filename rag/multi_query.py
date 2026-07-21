"""
Generate multiple search queries from a single user question.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.config import Config
from rag.logger import get_logger

logger = get_logger(__name__)


class MultiQueryGenerator:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Generate 3 different search queries for the user's question.

Rules:
- Keep same meaning.
- Different wording.
- One query per line.
- No numbering.
- No explanations.
""",
                ),
                (
                    "human",
                    "{question}",
                ),
            ]
        )

        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generate(
        self,
        question: str,
    ) -> list[str]:

        result = self.chain.invoke(
            {
                "question": question,
            }
        )

        queries = [
            q.strip()
            for q in result.splitlines()
            if q.strip()
        ]

        queries.insert(0, question)

        logger.info("Generated %d search queries.", len(queries))

        for query in queries:
            logger.info("• %s", query)

        return list(dict.fromkeys(queries))
