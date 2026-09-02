"""Tests for the Paddle webhook receipt endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.main import app
from rag.config import Config
from rag.models import BillingEvent


SECRET = "test_paddle_webhook_secret"
TIMESTAMP = 1_700_000_000

EVENT_PAYLOAD = {
    "event_id": "evt_123",
    "event_type": "subscription.updated",
    "data": {
        "id": "sub_123",
    },
}


class FakeSession:
    """Minimal database session fake for Paddle webhook tests."""

    def __init__(
        self,
        *,
        existing_event: BillingEvent | None = None,
        scalar_error: SQLAlchemyError | None = None,
        commit_error: SQLAlchemyError | None = None,
        scalar_results: list[BillingEvent | None] | None = None,
    ) -> None:
        self.existing_event = existing_event
        self.scalar_error = scalar_error
        self.commit_error = commit_error
        self.scalar_results = scalar_results

        self.scalar_calls = 0
        self.statements: list[Any] = []
        self.added: list[BillingEvent] = []
        self.commit_calls = 0
        self.rollback_calls = 0

    def scalar(
        self,
        statement: Any,
    ) -> BillingEvent | None:
        self.scalar_calls += 1
        self.statements.append(statement)

        if self.scalar_error is not None:
            raise self.scalar_error

        if self.scalar_results is not None:
            index = self.scalar_calls - 1

            if index < len(self.scalar_results):
                return self.scalar_results[index]

            return None

        return self.existing_event

    def add(
        self,
        value: BillingEvent,
    ) -> None:
        self.added.append(value)

    def commit(
        self,
    ) -> None:
        self.commit_calls += 1

        if self.commit_error is not None:
            raise self.commit_error

    def rollback(
        self,
    ) -> None:
        self.rollback_calls += 1


@pytest.fixture(autouse=True)
def clear_dependency_overrides(
) -> Iterator[None]:
    app.dependency_overrides.clear()

    original_secret = Config.PADDLE_WEBHOOK_SECRET
    original_tolerance = Config.PADDLE_WEBHOOK_TOLERANCE_SECONDS

    Config.PADDLE_WEBHOOK_SECRET = SECRET

    # Large tolerance keeps static test timestamps deterministic.
    Config.PADDLE_WEBHOOK_TOLERANCE_SECONDS = 10_000_000_000

    yield

    Config.PADDLE_WEBHOOK_SECRET = original_secret
    Config.PADDLE_WEBHOOK_TOLERANCE_SECONDS = original_tolerance

    app.dependency_overrides.clear()


def _override_database(
    db: FakeSession,
) -> None:
    app.dependency_overrides[
        get_db
    ] = lambda: cast(
        Session,
        db,
    )


def _raw_body(
    payload: Any = EVENT_PAYLOAD,
) -> bytes:
    return json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")


def _signature_header(
    raw_body: bytes,
    *,
    timestamp: int = TIMESTAMP,
    secret: str = SECRET,
) -> str:
    signed_payload = (
        str(timestamp).encode("ascii")
        + b":"
        + raw_body
    )

    signature = hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    return (
        f"ts={timestamp};"
        f"h1={signature}"
    )


def _post_webhook(
    client: TestClient,
    *,
    raw_body: bytes,
    signature_header: str | None = None,
):
    headers = {
        "Content-Type": "application/json",
    }

    if signature_header is not None:
        headers["Paddle-Signature"] = signature_header

    return client.post(
        "/webhooks/paddle",
        content=raw_body,
        headers=headers,
    )


def test_valid_webhook_is_persisted() -> None:
    db = FakeSession()

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body()

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "received",
    }

    assert db.scalar_calls == 1
    assert len(db.added) == 1

    event = db.added[0]

    assert event.provider == "paddle"
    assert event.provider_event_id == "evt_123"
    assert event.event_type == "subscription.updated"
    assert event.payload == EVENT_PAYLOAD

    assert event.organization_id is None
    assert event.subscription_id is None
    assert event.processed_at is None

    assert db.commit_calls == 1
    assert db.rollback_calls == 0


def test_duplicate_webhook_returns_success_without_insert() -> None:
    existing_event = BillingEvent(
        provider="paddle",
        provider_event_id="evt_123",
        event_type="subscription.updated",
        payload=EVENT_PAYLOAD,
    )

    db = FakeSession(
        existing_event=existing_event
    )

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body()

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "duplicate",
    }

    assert db.scalar_calls == 1
    assert db.added == []
    assert db.commit_calls == 0
    assert db.rollback_calls == 0


def test_missing_signature_is_rejected_before_database_access(
) -> None:
    db = FakeSession()

    _override_database(db)

    client = TestClient(app)

    response = _post_webhook(
        client,
        raw_body=_raw_body(),
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid Paddle webhook signature."
    }

    assert db.scalar_calls == 0
    assert db.added == []
    assert db.commit_calls == 0


def test_invalid_signature_is_rejected_before_database_access(
) -> None:
    db = FakeSession()

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body()

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=(
            f"ts={TIMESTAMP};"
            "h1=invalid-signature"
        ),
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid Paddle webhook signature."
    }

    assert db.scalar_calls == 0
    assert db.added == []
    assert db.commit_calls == 0


def test_valid_signature_with_invalid_json_returns_400(
) -> None:
    db = FakeSession()

    _override_database(db)

    client = TestClient(app)

    raw_body = b"{not-valid-json"

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid Paddle webhook payload."
    }

    assert db.scalar_calls == 0
    assert db.added == []
    assert db.commit_calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {
            "event_type": "subscription.updated",
        },
        {
            "event_id": "evt_123",
        },
        {
            "event_id": "",
            "event_type": "subscription.updated",
        },
        {
            "event_id": "evt_123",
            "event_type": "",
        },
        {
            "event_id": 123,
            "event_type": "subscription.updated",
        },
    ],
)
def test_invalid_event_envelope_returns_400(
    payload: Any,
) -> None:
    db = FakeSession()

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body(
        payload
    )

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid Paddle webhook payload."
    }

    assert db.scalar_calls == 0
    assert db.added == []
    assert db.commit_calls == 0


def test_idempotency_query_failure_returns_safe_500(
) -> None:
    db = FakeSession(
        scalar_error=SQLAlchemyError(
            "Database unavailable."
        )
    )

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body()

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Unable to process Paddle webhook."
    }

    assert db.added == []
    assert db.commit_calls == 0
    assert db.rollback_calls == 1


def test_commit_database_failure_returns_safe_500(
) -> None:
    db = FakeSession(
        commit_error=SQLAlchemyError(
            "Commit failed."
        )
    )

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body()

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Unable to process Paddle webhook."
    }

    assert len(db.added) == 1
    assert db.commit_calls == 1
    assert db.rollback_calls == 1


def test_concurrent_duplicate_integrity_error_is_idempotent(
) -> None:
    existing_event = BillingEvent(
        provider="paddle",
        provider_event_id="evt_123",
        event_type="subscription.updated",
        payload=EVENT_PAYLOAD,
    )

    db = FakeSession(
        commit_error=IntegrityError(
            "INSERT",
            {},
            Exception("duplicate"),
        ),
        scalar_results=[
            None,
            existing_event,
        ],
    )

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body()

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "duplicate",
    }

    assert db.scalar_calls == 2
    assert len(db.added) == 1
    assert db.commit_calls == 1
    assert db.rollback_calls == 1


def test_unrelated_integrity_error_returns_safe_500(
) -> None:
    db = FakeSession(
        commit_error=IntegrityError(
            "INSERT",
            {},
            Exception("constraint failure"),
        ),
        scalar_results=[
            None,
            None,
        ],
    )

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body()

    response = _post_webhook(
        client,
        raw_body=raw_body,
        signature_header=_signature_header(
            raw_body
        ),
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Unable to process Paddle webhook."
    }

    assert db.scalar_calls == 2
    assert len(db.added) == 1
    assert db.commit_calls == 1
    assert db.rollback_calls == 1
