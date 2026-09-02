"""Tests for the Paddle webhook processing endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

import api.paddle_webhook_routes as webhook_routes
from api.dependencies import get_db
from api.main import app
from rag.config import Config
from rag.models import BillingEvent
from rag.paddle_subscription_service import (
    PaddleSubscriptionServiceError,
)
from rag.paddle_subscription_sync import (
    PaddleSubscriptionState,
    PaddleSubscriptionSyncError,
)


SECRET = "test_paddle_webhook_secret"
TIMESTAMP = 1_700_000_000

PROCESSED_AT = datetime(
    2026,
    9,
    2,
    10,
    0,
    tzinfo=timezone.utc,
)


SUBSCRIPTION_PAYLOAD = {
    "event_id": "evt_123",
    "event_type": "subscription.updated",
    "data": {
        "id": "sub_123",
    },
}


UNSUPPORTED_PAYLOAD = {
    "event_id": "evt_transaction",
    "event_type": "transaction.completed",
    "data": {
        "id": "txn_123",
    },
}


class FakeSession:
    """Minimal database session fake for Paddle webhook route tests."""

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
        self.statements.append(
            statement
        )

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
        self.added.append(
            value
        )

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
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    app.dependency_overrides.clear()

    original_secret = Config.PADDLE_WEBHOOK_SECRET

    original_tolerance = (
        Config.PADDLE_WEBHOOK_TOLERANCE_SECONDS
    )

    Config.PADDLE_WEBHOOK_SECRET = SECRET

    # Large tolerance keeps static webhook timestamps deterministic.
    Config.PADDLE_WEBHOOK_TOLERANCE_SECONDS = (
        10_000_000_000
    )

    monkeypatch.setattr(
        webhook_routes,
        "_utc_now",
        lambda: PROCESSED_AT,
    )

    yield

    Config.PADDLE_WEBHOOK_SECRET = original_secret

    Config.PADDLE_WEBHOOK_TOLERANCE_SECONDS = (
        original_tolerance
    )

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
    payload: Any = SUBSCRIPTION_PAYLOAD,
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


def _processed_event(
    *,
    event_id: str = "evt_123",
) -> BillingEvent:
    return BillingEvent(
        provider="paddle",
        provider_event_id=event_id,
        event_type="subscription.updated",
        payload=SUBSCRIPTION_PAYLOAD,
        processed_at=PROCESSED_AT,
    )


def _unprocessed_event(
    *,
    event_id: str = "evt_123",
) -> BillingEvent:
    return BillingEvent(
        provider="paddle",
        provider_event_id=event_id,
        event_type="subscription.updated",
        payload=SUBSCRIPTION_PAYLOAD,
        processed_at=None,
    )


def _fake_state() -> PaddleSubscriptionState:
    return Mock(
        spec=PaddleSubscriptionState
    )


def _patch_subscription_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, Mock]:
    parser = Mock(
        return_value=_fake_state()
    )

    synchronize = Mock()

    service_instance = Mock()
    service_instance.synchronize = synchronize

    service_factory = Mock(
        return_value=service_instance
    )

    monkeypatch.setattr(
        webhook_routes,
        "parse_paddle_subscription_event",
        parser,
    )

    monkeypatch.setattr(
        webhook_routes,
        "PaddleSubscriptionService",
        service_factory,
    )

    return parser, synchronize


def test_supported_subscription_webhook_is_processed_and_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession()

    parser, synchronize = (
        _patch_subscription_processing(
            monkeypatch
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
    assert event.payload == SUBSCRIPTION_PAYLOAD

    parser.assert_called_once_with(
        SUBSCRIPTION_PAYLOAD
    )

    synchronize.assert_called_once()

    call = synchronize.call_args

    assert call.kwargs["billing_event"] is event

    assert db.commit_calls == 1
    assert db.rollback_calls == 0


def test_unsupported_webhook_is_persisted_and_marked_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession()

    parser = Mock()

    monkeypatch.setattr(
        webhook_routes,
        "parse_paddle_subscription_event",
        parser,
    )

    _override_database(db)

    client = TestClient(app)

    raw_body = _raw_body(
        UNSUPPORTED_PAYLOAD
    )

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

    assert len(db.added) == 1

    event = db.added[0]

    assert event.event_type == "transaction.completed"
    assert event.processed_at == PROCESSED_AT

    parser.assert_not_called()

    assert db.commit_calls == 1
    assert db.rollback_calls == 0


def test_processed_duplicate_returns_success_without_reprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_event = _processed_event()

    db = FakeSession(
        existing_event=existing_event
    )

    parser, synchronize = (
        _patch_subscription_processing(
            monkeypatch
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

    assert response.status_code == 200

    assert response.json() == {
        "status": "duplicate",
    }

    parser.assert_not_called()
    synchronize.assert_not_called()

    assert db.added == []
    assert db.commit_calls == 0
    assert db.rollback_calls == 0


def test_unprocessed_existing_event_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_event = _unprocessed_event()

    db = FakeSession(
        existing_event=existing_event
    )

    parser, synchronize = (
        _patch_subscription_processing(
            monkeypatch
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

    assert response.status_code == 200

    assert response.json() == {
        "status": "received",
    }

    parser.assert_called_once_with(
        existing_event.payload
    )

    synchronize.assert_called_once()

    assert (
        synchronize.call_args.kwargs["billing_event"]
        is existing_event
    )

    assert db.added == []
    assert db.commit_calls == 1
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


def test_subscription_parser_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession()

    monkeypatch.setattr(
        webhook_routes,
        "parse_paddle_subscription_event",
        Mock(
            side_effect=PaddleSubscriptionSyncError(
                "Invalid subscription event."
            )
        ),
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
    assert db.commit_calls == 0
    assert db.rollback_calls == 1


def test_subscription_service_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession()

    parser, synchronize = (
        _patch_subscription_processing(
            monkeypatch
        )
    )

    synchronize.side_effect = (
        PaddleSubscriptionServiceError(
            "Unable to synchronize."
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

    parser.assert_called_once()

    assert len(db.added) == 1
    assert db.commit_calls == 0
    assert db.rollback_calls == 1


def test_commit_database_failure_returns_safe_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession(
        commit_error=SQLAlchemyError(
            "Commit failed."
        )
    )

    _patch_subscription_processing(
        monkeypatch
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


def test_concurrent_processed_duplicate_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_event = _processed_event()

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

    _patch_subscription_processing(
        monkeypatch
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
    monkeypatch: pytest.MonkeyPatch,
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

    _patch_subscription_processing(
        monkeypatch
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
