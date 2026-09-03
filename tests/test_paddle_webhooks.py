"""Tests for Paddle webhook signature verification."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from rag.paddle_webhooks import (
    PaddleWebhookVerificationError,
    verify_paddle_webhook,
)


SECRET = "test_webhook_secret"
TIMESTAMP = 1_700_000_000
RAW_BODY = b'{"event_id":"evt_123","event_type":"subscription.updated"}'


def _signature(
    *,
    body: bytes = RAW_BODY,
    timestamp: int = TIMESTAMP,
    secret: str = SECRET,
) -> str:
    signed_payload = (
        str(timestamp).encode("ascii")
        + b":"
        + body
    )

    return hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()


def test_valid_signature_is_accepted() -> None:
    signature = _signature()

    verify_paddle_webhook(
        raw_body=RAW_BODY,
        signature_header=(
            f"ts={TIMESTAMP};h1={signature}"
        ),
        secret=SECRET,
        tolerance_seconds=5,
        current_time=lambda: float(TIMESTAMP),
    )


def test_tampered_body_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="Invalid Paddle webhook signature",
    ):
        verify_paddle_webhook(
            raw_body=b'{"tampered":true}',
            signature_header=(
                f"ts={TIMESTAMP};h1={signature}"
            ),
            secret=SECRET,
            tolerance_seconds=5,
            current_time=lambda: float(TIMESTAMP),
        )


def test_wrong_secret_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="Invalid Paddle webhook signature",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=(
                f"ts={TIMESTAMP};h1={signature}"
            ),
            secret="wrong_secret",
            tolerance_seconds=5,
            current_time=lambda: float(TIMESTAMP),
        )


def test_missing_timestamp_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="timestamp is missing",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=f"h1={signature}",
            secret=SECRET,
            current_time=lambda: float(TIMESTAMP),
        )


def test_invalid_timestamp_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="Invalid Paddle webhook timestamp",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=(
                f"ts=not-an-integer;h1={signature}"
            ),
            secret=SECRET,
            current_time=lambda: float(TIMESTAMP),
        )


def test_missing_signature_is_rejected() -> None:
    with pytest.raises(
        PaddleWebhookVerificationError,
        match="signature is missing",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=f"ts={TIMESTAMP}",
            secret=SECRET,
            current_time=lambda: float(TIMESTAMP),
        )


def test_empty_secret_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="secret is not configured",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=(
                f"ts={TIMESTAMP};h1={signature}"
            ),
            secret="",
            current_time=lambda: float(TIMESTAMP),
        )


def test_negative_tolerance_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="tolerance must not be negative",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=(
                f"ts={TIMESTAMP};h1={signature}"
            ),
            secret=SECRET,
            tolerance_seconds=-1,
            current_time=lambda: float(TIMESTAMP),
        )


def test_stale_webhook_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="outside the allowed tolerance",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=(
                f"ts={TIMESTAMP};h1={signature}"
            ),
            secret=SECRET,
            tolerance_seconds=5,
            current_time=lambda: float(
                TIMESTAMP + 6
            ),
        )


def test_future_webhook_outside_tolerance_is_rejected() -> None:
    signature = _signature()

    with pytest.raises(
        PaddleWebhookVerificationError,
        match="outside the allowed tolerance",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=(
                f"ts={TIMESTAMP};h1={signature}"
            ),
            secret=SECRET,
            tolerance_seconds=5,
            current_time=lambda: float(
                TIMESTAMP - 6
            ),
        )


def test_timestamp_at_tolerance_boundary_is_accepted() -> None:
    signature = _signature()

    verify_paddle_webhook(
        raw_body=RAW_BODY,
        signature_header=(
            f"ts={TIMESTAMP};h1={signature}"
        ),
        secret=SECRET,
        tolerance_seconds=5,
        current_time=lambda: float(
            TIMESTAMP + 5
        ),
    )


def test_one_valid_signature_among_multiple_is_accepted() -> None:
    valid_signature = _signature()

    verify_paddle_webhook(
        raw_body=RAW_BODY,
        signature_header=(
            f"ts={TIMESTAMP};"
            "h1=invalid-signature;"
            f"h1={valid_signature}"
        ),
        secret=SECRET,
        tolerance_seconds=5,
        current_time=lambda: float(TIMESTAMP),
    )


def test_multiple_invalid_signatures_are_rejected() -> None:
    with pytest.raises(
        PaddleWebhookVerificationError,
        match="Invalid Paddle webhook signature",
    ):
        verify_paddle_webhook(
            raw_body=RAW_BODY,
            signature_header=(
                f"ts={TIMESTAMP};"
                "h1=invalid-one;"
                "h1=invalid-two"
            ),
            secret=SECRET,
            tolerance_seconds=5,
            current_time=lambda: float(TIMESTAMP),
        )


def test_unknown_signature_components_are_ignored() -> None:
    signature = _signature()

    verify_paddle_webhook(
        raw_body=RAW_BODY,
        signature_header=(
            f"ts={TIMESTAMP};"
            "v=1;"
            "foo=bar;"
            f"h1={signature}"
        ),
        secret=SECRET,
        tolerance_seconds=5,
        current_time=lambda: float(TIMESTAMP),
    )
