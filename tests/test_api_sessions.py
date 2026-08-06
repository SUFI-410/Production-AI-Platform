from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.routes as routes_module
from api.main import app


@dataclass
class FakeApplication:
    """
    Record arguments received from the FastAPI route.
    """

    calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def ask_with_sources(
        self,
        question: str,
        session_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "question": question,
                "session_id": session_id,
                "use_cache": use_cache,
            }
        )

        resolved_session_id = (
            session_id
            if session_id is not None
            else "server-generated-session"
        )

        return {
            "answer": f"answer::{question}",
            "sources": [
                {
                    "document": "python_oop.md",
                    "score": 0.99,
                    "metadata": {},
                }
            ],
            "session_id": resolved_session_id,
            "cached": False,
            "grounded": True,
        }


def test_chat_forwards_session_id_and_use_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_application = FakeApplication()
    verification_calls: list[dict[str, str]] = []

    def fake_verify_turnstile(
        token: str,
        request: Any,
    ) -> None:
        verification_calls.append(
            {
                "token": token,
                "path": request.url.path,
            }
        )

    monkeypatch.setattr(
        routes_module,
        "verify_turnstile",
        fake_verify_turnstile,
    )

    monkeypatch.setattr(
        routes_module,
        "get_application",
        lambda: fake_application,
    )

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "question": "Explain inheritance.",
            "turnstile_token": "valid-test-token",
            "session_id": "session-123",
            "use_cache": False,
        },
    )

    assert response.status_code == 200

    assert verification_calls == [
        {
            "token": "valid-test-token",
            "path": "/chat",
        }
    ]

    assert fake_application.calls == [
        {
            "question": "Explain inheritance.",
            "session_id": "session-123",
            "use_cache": False,
        }
    ]

    body = response.json()

    assert body["answer"] == (
        "answer::Explain inheritance."
    )
    assert body["session_id"] == "session-123"
    assert body["cached"] is False
    assert body["grounded"] is True

    assert body["sources"] == [
        {
            "document": "python_oop.md",
            "score": 0.99,
            "metadata": {},
        }
    ]

    assert body["latency_ms"] >= 0


def test_chat_returns_generated_session_id_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_application = FakeApplication()

    monkeypatch.setattr(
        routes_module,
        "verify_turnstile",
        lambda token, request: None,
    )

    monkeypatch.setattr(
        routes_module,
        "get_application",
        lambda: fake_application,
    )

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "question": "What is polymorphism?",
            "turnstile_token": "valid-test-token",
        },
    )

    assert response.status_code == 200

    assert fake_application.calls == [
        {
            "question": "What is polymorphism?",
            "session_id": None,
            "use_cache": True,
        }
    ]

    body = response.json()

    assert body["session_id"] == (
        "server-generated-session"
    )
    assert body["answer"] == (
        "answer::What is polymorphism?"
    )
