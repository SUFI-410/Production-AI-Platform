from __future__ import annotations

from typing import Any, cast

from langchain_core.documents import Document

from rag.chain import (
    INSUFFICIENT_CONTEXT_ANSWER,
    NO_RELEVANT_INFORMATION_ANSWER,
    RAGChain,
)


class FakeRetriever:
    @staticmethod
    def sources(
        documents: list[Document],
    ) -> list[dict[str, object]]:
        return [
            {
                "document": "test-document.md",
                "score": 1.0,
            }
            for _ in documents
        ]


def build_chain(
    answer: str,
    generated_documents: list[Document],
) -> RAGChain:
    chain = cast(
        Any,
        object.__new__(RAGChain),
    )

    chain.retriever = FakeRetriever()

    def fake_generate(
        question: str,
        documents: list[Document],
        history: str = "",
    ) -> tuple[str, list[Document]]:
        del question, documents, history

        return answer, generated_documents

    chain._generate = fake_generate

    return chain


def test_supported_answer_is_grounded() -> None:
    document = Document(
        page_content=(
            "Inheritance allows a child class to "
            "acquire behavior from a parent class."
        )
    )

    chain = build_chain(
        answer="Inheritance uses a parent class.",
        generated_documents=[document],
    )

    result = chain.ask(
        question="What is inheritance?",
        documents=[document],
    )

    assert result["grounded"] is True
    assert result["documents"] == [document]
    assert len(result["sources"]) == 1


def test_no_documents_response_is_not_grounded() -> None:
    chain = build_chain(
        answer=NO_RELEVANT_INFORMATION_ANSWER,
        generated_documents=[],
    )

    result = chain.ask(
        question="Who is Sufyan?",
        documents=[],
    )

    assert result["grounded"] is False
    assert result["documents"] == []
    assert result["sources"] == []


def test_refusal_with_documents_is_not_grounded() -> None:
    document = Document(
        page_content="Inheritance is an OOP feature."
    )

    chain = build_chain(
        answer=INSUFFICIENT_CONTEXT_ANSWER,
        generated_documents=[document],
    )

    result = chain.ask(
        question=(
            "How is inheritance different "
            "from composition?"
        ),
        documents=[document],
    )

    assert result["grounded"] is False
    assert result["documents"] == []
    assert result["sources"] == []
