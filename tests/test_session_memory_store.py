from __future__ import annotations

import threading

import pytest
from pytest import MonkeyPatch

import rag.memory as memory_module
from rag.memory import (
    SessionCapacityError,
    SessionMemoryStore,
)


class FakeClock:
    def __init__(self) -> None:
        self.current = 1000.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def join_thread(thread: threading.Thread) -> None:
    thread.join(timeout=2)

    assert not thread.is_alive(), (
        f"Thread {thread.name!r} did not finish."
    )


def test_sessions_have_isolated_memory() -> None:
    store = SessionMemoryStore(
        ttl_seconds=60,
        max_sessions=10,
        max_messages=10,
    )

    with store.session("user-a") as memory:
        memory.add_exchange(
            "Question from user A",
            "Answer for user A",
        )

    with store.session("user-b") as memory:
        assert memory.get_messages() == []

        memory.add_exchange(
            "Question from user B",
            "Answer for user B",
        )

    with store.session("user-a") as memory:
        assert memory.get_messages() == [
            {
                "role": "user",
                "content": "Question from user A",
            },
            {
                "role": "assistant",
                "content": "Answer for user A",
            },
        ]

    with store.session("user-b") as memory:
        assert memory.get_messages() == [
            {
                "role": "user",
                "content": "Question from user B",
            },
            {
                "role": "assistant",
                "content": "Answer for user B",
            },
        ]


def test_session_expires_after_ttl(
    monkeypatch: MonkeyPatch,
) -> None:
    clock = FakeClock()

    monkeypatch.setattr(
        memory_module.time,
        "monotonic",
        clock,
    )

    store = SessionMemoryStore(
        ttl_seconds=10,
        max_sessions=10,
        max_messages=10,
    )

    with store.session("expiring-session") as memory:
        memory.add_exchange(
            "Original question",
            "Original answer",
        )

    assert store.size() == 1

    clock.advance(10.1)

    assert store.size() == 0

    with store.session("expiring-session") as memory:
        assert memory.get_messages() == []


def test_oldest_inactive_session_is_evicted(
    monkeypatch: MonkeyPatch,
) -> None:
    clock = FakeClock()

    monkeypatch.setattr(
        memory_module.time,
        "monotonic",
        clock,
    )

    store = SessionMemoryStore(
        ttl_seconds=60,
        max_sessions=2,
        max_messages=10,
    )

    with store.session("oldest") as memory:
        memory.add_exchange(
            "Old question",
            "Old answer",
        )

    clock.advance(1)

    with store.session("newer") as memory:
        memory.add_exchange(
            "Newer question",
            "Newer answer",
        )

    clock.advance(1)

    with store.session("newest") as memory:
        memory.add_exchange(
            "Newest question",
            "Newest answer",
        )

    assert store.size() == 2

    with store.session("newer") as memory:
        assert len(memory) == 2

    with store.session("oldest") as memory:
        assert memory.get_messages() == []


def test_active_session_is_not_evicted() -> None:
    store = SessionMemoryStore(
        ttl_seconds=60,
        max_sessions=1,
        max_messages=10,
    )

    session_entered = threading.Event()
    release_session = threading.Event()
    errors: list[BaseException] = []

    def hold_session() -> None:
        try:
            with store.session("active-session"):
                session_entered.set()

                if not release_session.wait(timeout=2):
                    raise TimeoutError(
                        "Timed out waiting to release session."
                    )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=hold_session,
        name="active-session-holder",
    )
    thread.start()

    try:
        assert session_entered.wait(timeout=1)

        with pytest.raises(SessionCapacityError):
            with store.session("another-session"):
                pass
    finally:
        release_session.set()
        join_thread(thread)

    assert errors == []


def test_same_session_requests_are_serialized() -> None:
    store = SessionMemoryStore(
        ttl_seconds=60,
        max_sessions=10,
        max_messages=10,
    )

    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []

    def first_request() -> None:
        try:
            with store.session("shared-session"):
                first_entered.set()

                if not release_first.wait(timeout=2):
                    raise TimeoutError(
                        "Timed out waiting to release first request."
                    )
        except BaseException as exc:
            errors.append(exc)

    def second_request() -> None:
        try:
            second_attempting.set()

            with store.session("shared-session"):
                second_entered.set()
        except BaseException as exc:
            errors.append(exc)

    first_thread = threading.Thread(
        target=first_request,
        name="first-shared-request",
    )
    second_thread = threading.Thread(
        target=second_request,
        name="second-shared-request",
    )

    first_thread.start()

    try:
        assert first_entered.wait(timeout=1)

        second_thread.start()

        assert second_attempting.wait(timeout=1)

        assert not second_entered.wait(timeout=0.2)

        release_first.set()

        assert second_entered.wait(timeout=1)
    finally:
        release_first.set()
        join_thread(first_thread)

        if second_thread.ident is not None:
            join_thread(second_thread)

    assert errors == []


def test_different_sessions_can_run_concurrently() -> None:
    store = SessionMemoryStore(
        ttl_seconds=60,
        max_sessions=10,
        max_messages=10,
    )

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []

    def first_request() -> None:
        try:
            with store.session("session-a"):
                first_entered.set()

                if not release_first.wait(timeout=2):
                    raise TimeoutError(
                        "Timed out waiting to release session A."
                    )
        except BaseException as exc:
            errors.append(exc)

    def second_request() -> None:
        try:
            with store.session("session-b"):
                second_entered.set()
        except BaseException as exc:
            errors.append(exc)

    first_thread = threading.Thread(
        target=first_request,
        name="session-a-request",
    )
    second_thread = threading.Thread(
        target=second_request,
        name="session-b-request",
    )

    first_thread.start()

    try:
        assert first_entered.wait(timeout=1)

        second_thread.start()

        assert second_entered.wait(timeout=1)
    finally:
        release_first.set()
        join_thread(first_thread)

        if second_thread.ident is not None:
            join_thread(second_thread)

    assert errors == []
