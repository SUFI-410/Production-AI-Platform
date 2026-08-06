from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from rag.application import RAGApplication
from rag.memory import SessionMemoryStore
from rag.response_cache import ResponseCache


@dataclass
class FakeQueryRewriter:
    calls: list[tuple[str, str]] = field(
        default_factory=list
    )

    def rewrite(
        self,
        question: str,
        history: str,
    ) -> str:
        self.calls.append(
            (
                question,
                history,
            )
        )

        return f"rewritten::{question}"


@dataclass
class FakeMultiQueryGenerator:
    calls: list[str] = field(
        default_factory=list
    )

    def generate(
        self,
        question: str,
    ) -> list[str]:
        self.calls.append(question)

        return [question]


@dataclass
class FakeChain:
    retrieve_calls: list[list[str]] = field(
        default_factory=list
    )

    ask_calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def retrieve(
        self,
        questions: list[str],
    ) -> list[str]:
        self.retrieve_calls.append(questions)

        return [
            f"document::{questions[0]}",
        ]

    def ask(
        self,
        question: str,
        documents: list[str],
        history: str = "",
    ) -> dict[str, Any]:
        self.ask_calls.append(
            {
                "question": question,
                "documents": documents,
                "history": history,
            }
        )

        return {
            "question": question,
            "answer": f"answer::{question}",
            "documents": documents,
            "sources": [
                {
                    "document": "test-document.md",
                    "score": 1.0,
                }
            ],
        }


def build_application() -> tuple[
    RAGApplication,
    FakeChain,
    FakeQueryRewriter,
]:
    """
    Build RAGApplication without initializing production models.

    The application is cast to Any because this test deliberately
    injects lightweight fakes in place of production dependencies.
    """

    application = cast(
        Any,
        object.__new__(RAGApplication),
    )

    chain = FakeChain()
    query_rewriter = FakeQueryRewriter()

    application.memory_store = SessionMemoryStore(
        ttl_seconds=60,
        max_sessions=20,
        max_messages=10,
    )

    application._default_session_id = (
        "test-default-session"
    )

    application.query_rewriter = query_rewriter

    application.multi_query = (
        FakeMultiQueryGenerator()
    )

    application.cache = ResponseCache(
        ttl_seconds=60
    )

    application.chain = chain

    return application, chain, query_rewriter


def test_conversation_histories_are_isolated() -> None:
    application, chain, query_rewriter = (
        build_application()
    )

    first_a = application.ask_with_sources(
        question="First question from A",
        session_id="session-a",
        use_cache=False,
    )

    first_b = application.ask_with_sources(
        question="First question from B",
        session_id="session-b",
        use_cache=False,
    )

    second_a = application.ask_with_sources(
        question="Follow-up from A",
        session_id="session-a",
        use_cache=False,
    )

    assert first_a["answer"] == (
        "answer::First question from A"
    )

    assert first_b["answer"] == (
        "answer::First question from B"
    )

    assert second_a["answer"] == (
        "answer::rewritten::Follow-up from A"
    )

    assert first_a["session_id"] == "session-a"
    assert first_b["session_id"] == "session-b"
    assert second_a["session_id"] == "session-a"

    assert len(query_rewriter.calls) == 1

    rewritten_question, history = (
        query_rewriter.calls[0]
    )

    assert rewritten_question == "Follow-up from A"

    assert "First question from A" in history
    assert "answer::First question from A" in history

    assert "First question from B" not in history
    assert "answer::First question from B" not in history

    assert chain.ask_calls[0]["history"] == ""
    assert chain.ask_calls[1]["history"] == ""

    second_a_history = chain.ask_calls[2]["history"]

    assert "First question from A" in second_a_history
    assert "First question from B" not in second_a_history

    with application.memory_store.session(
        "session-a"
    ) as memory:
        session_a_messages = memory.get_messages()

    with application.memory_store.session(
        "session-b"
    ) as memory:
        session_b_messages = memory.get_messages()

    assert len(session_a_messages) == 4
    assert len(session_b_messages) == 2

    assert session_a_messages[0]["content"] == (
        "First question from A"
    )

    assert session_b_messages[0]["content"] == (
        "First question from B"
    )


def test_cache_keys_are_session_and_history_aware() -> None:
    application, _, _ = build_application()

    session_a_key = application._cache_key(
        question="What is it?",
        metadata_filter=None,
        session_id="session-a",
        history="",
    )

    session_b_key = application._cache_key(
        question="What is it?",
        metadata_filter=None,
        session_id="session-b",
        history="",
    )

    later_history_key = application._cache_key(
        question="What is it?",
        metadata_filter=None,
        session_id="session-a",
        history="User: Earlier question",
    )

    assert session_a_key != session_b_key
    assert session_a_key != later_history_key


def test_cache_hit_updates_session_memory() -> None:
    application, chain, _ = build_application()

    cache_key = application._cache_key(
        question="Cached question",
        metadata_filter=None,
        session_id="cache-session",
        history="",
    )

    application.cache.set(
        cache_key,
        {
            "question": "Cached question",
            "answer": "Cached answer",
            "documents": [],
            "sources": [],
            "cached": False,
        },
    )

    result = application.ask_with_sources(
        question="Cached question",
        session_id="cache-session",
        use_cache=True,
    )

    assert result["answer"] == "Cached answer"
    assert result["cached"] is True
    assert result["session_id"] == "cache-session"

    assert chain.retrieve_calls == []
    assert chain.ask_calls == []

    with application.memory_store.session(
        "cache-session"
    ) as memory:
        assert memory.get_messages() == [
            {
                "role": "user",
                "content": "Cached question",
            },
            {
                "role": "assistant",
                "content": "Cached answer",
            },
        ]

def test_repeated_standalone_question_uses_cache() -> None:
    application, chain, query_rewriter = (
        build_application()
    )

    first_result = application.ask_with_sources(
        question="What is inheritance in Python?",
        session_id="repeat-session",
        use_cache=True,
    )

    second_result = application.ask_with_sources(
        question="What is inheritance in Python?",
        session_id="repeat-session",
        use_cache=True,
    )

    assert first_result["cached"] is False
    assert second_result["cached"] is True

    assert second_result["answer"] == (
        first_result["answer"]
    )

    assert len(chain.retrieve_calls) == 1
    assert len(chain.ask_calls) == 1

    assert query_rewriter.calls == []

    with application.memory_store.session(
        "repeat-session"
    ) as memory:
        messages = memory.get_messages()

    assert len(messages) == 4


def test_contextual_question_remains_history_aware() -> None:
    application, chain, query_rewriter = (
        build_application()
    )

    application.ask_with_sources(
        question="What is inheritance in Python?",
        session_id="context-session",
        use_cache=False,
    )

    first_follow_up = application.ask_with_sources(
        question="Give me a simple example of it?",
        session_id="context-session",
        use_cache=True,
    )

    application.ask_with_sources(
        question="What is a decorator in Python?",
        session_id="context-session",
        use_cache=False,
    )

    second_follow_up = application.ask_with_sources(
        question="Give me a simple example of it?",
        session_id="context-session",
        use_cache=True,
    )

    assert first_follow_up["cached"] is False
    assert second_follow_up["cached"] is False

    assert len(chain.retrieve_calls) == 4
    assert len(chain.ask_calls) == 4

    assert len(query_rewriter.calls) == 2

    first_history = query_rewriter.calls[0][1]
    second_history = query_rewriter.calls[1][1]

    assert first_history != second_history
    assert "inheritance" in first_history
    assert "decorator" in second_history


def test_use_cache_false_bypasses_cached_response() -> None:
    application, chain, _ = build_application()

    cache_key = application._cache_key(
        question="Do not use cache",
        metadata_filter=None,
        session_id="no-cache-session",
        history="",
    )

    application.cache.set(
        cache_key,
        {
            "question": "Do not use cache",
            "answer": "Old cached answer",
            "documents": [],
            "sources": [],
            "cached": False,
        },
    )

    result = application.ask_with_sources(
        question="Do not use cache",
        session_id="no-cache-session",
        use_cache=False,
    )

    assert result["answer"] == (
        "answer::Do not use cache"
    )

    assert result["cached"] is False
    assert result["session_id"] == "no-cache-session"

    assert len(chain.retrieve_calls) == 1
    assert len(chain.ask_calls) == 1

    cached_result = application.cache.get(
        cache_key
    )

    assert cached_result is not None

    assert cached_result["answer"] == (
        "Old cached answer"
    )


def test_calls_without_session_id_share_private_default_session() -> None:
    application, _, query_rewriter = (
        build_application()
    )

    first_result = application.ask_with_sources(
        question="Initial local question",
        use_cache=False,
    )

    second_result = application.ask_with_sources(
        question="Local follow-up",
        use_cache=False,
    )

    assert first_result["session_id"] == (
        "test-default-session"
    )

    assert second_result["session_id"] == (
        "test-default-session"
    )

    assert len(query_rewriter.calls) == 1

    question, history = query_rewriter.calls[0]

    assert question == "Local follow-up"
    assert "Initial local question" in history
