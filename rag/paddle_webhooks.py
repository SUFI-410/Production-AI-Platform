"""
Paddle webhook signature verification.

Webhook bodies must be verified using the exact raw bytes received from
Paddle before JSON parsing or subscription state changes occur.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Callable


class PaddleWebhookVerificationError(RuntimeError):
    """Raised when a Paddle webhook signature cannot be verified."""


@dataclass(frozen=True)
class PaddleSignature:
    """Parsed Paddle-Signature header values."""

    timestamp: int
    signatures: tuple[str, ...]


def _parse_signature_header(
    header: str,
) -> PaddleSignature:
    """Parse Paddle-Signature into timestamp and h1 signatures."""

    timestamp: int | None = None
    signatures: list[str] = []

    for component in header.split(";"):
        key, separator, value = component.partition("=")

        if not separator:
            continue

        key = key.strip()
        value = value.strip()

        if key == "ts":
            try:
                timestamp = int(value)
            except ValueError as exc:
                raise PaddleWebhookVerificationError(
                    "Invalid Paddle webhook timestamp."
                ) from exc

        elif key == "h1" and value:
            signatures.append(value)

    if timestamp is None:
        raise PaddleWebhookVerificationError(
            "Paddle webhook timestamp is missing."
        )

    if not signatures:
        raise PaddleWebhookVerificationError(
            "Paddle webhook signature is missing."
        )

    return PaddleSignature(
        timestamp=timestamp,
        signatures=tuple(signatures),
    )


def verify_paddle_webhook(
    *,
    raw_body: bytes,
    signature_header: str,
    secret: str,
    tolerance_seconds: int = 5,
    current_time: Callable[[], float] = time.time,
) -> None:
    """
    Verify a Paddle webhook signature and replay-protection timestamp.

    Raises PaddleWebhookVerificationError when verification fails.
    """

    if not secret:
        raise PaddleWebhookVerificationError(
            "Paddle webhook secret is not configured."
        )

    if tolerance_seconds < 0:
        raise PaddleWebhookVerificationError(
            "Paddle webhook tolerance must not be negative."
        )

    parsed = _parse_signature_header(
        signature_header
    )

    now = int(current_time())

    if abs(now - parsed.timestamp) > tolerance_seconds:
        raise PaddleWebhookVerificationError(
            "Paddle webhook timestamp is outside the allowed tolerance."
        )

    signed_payload = (
        str(parsed.timestamp).encode("ascii")
        + b":"
        + raw_body
    )

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    if not any(
        hmac.compare_digest(
            expected_signature,
            signature,
        )
        for signature in parsed.signatures
    ):
        raise PaddleWebhookVerificationError(
            "Invalid Paddle webhook signature."
        )
