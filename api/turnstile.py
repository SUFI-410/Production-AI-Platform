"""
Cloudflare Turnstile server-side verification.

The private widget secret is read exclusively from the
TURNSTILE_SECRET environment variable.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import HTTPException, Request, status

from rag.logger import get_logger

logger = get_logger(__name__)

SITEVERIFY_URL = (
    "https://challenges.cloudflare.com/"
    "turnstile/v0/siteverify"
)

SITEVERIFY_TIMEOUT_SECONDS = 10.0

CONFIGURATION_ERROR_CODES = {
    "invalid-input-secret",
    "missing-input-secret",
}

TRANSIENT_ERROR_CODES = {
    "internal-error",
}


def _client_ip(request: Request) -> str:
    """
    Return the original visitor IP when available.
    """

    cloudflare_ip = request.headers.get(
        "CF-Connecting-IP"
    )

    if cloudflare_ip:
        return cloudflare_ip.strip()

    forwarded_for = request.headers.get(
        "X-Forwarded-For"
    )

    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()

    if request.client is not None:
        return request.client.host

    return ""


def _error_codes(
    result: dict[str, Any],
) -> list[str]:
    """
    Return sanitized Siteverify error codes.
    """

    raw_error_codes = result.get(
        "error-codes",
        [],
    )

    if not isinstance(raw_error_codes, list):
        return []

    return [
        error_code
        for error_code in raw_error_codes
        if isinstance(error_code, str)
    ]


def _verification_unavailable() -> HTTPException:
    """
    Return an error for configuration or upstream failures.
    """

    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "Human verification is temporarily "
            "unavailable."
        ),
    )


def _verification_rejected() -> HTTPException:
    """
    Return an error for an invalid or expired token.
    """

    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Human verification failed.",
    )


def verify_turnstile(
    token: str,
    request: Request,
) -> None:
    """
    Validate a single-use Turnstile token.

    Verification fails closed when the secret is missing,
    Cloudflare cannot be reached, the response is invalid,
    or Cloudflare returns success=false.
    """

    secret = os.getenv("TURNSTILE_SECRET")

    if not secret:
        logger.error(
            "TURNSTILE_SECRET is not configured."
        )

        raise _verification_unavailable()

    payload = {
        "secret": secret,
        "response": token,
        "remoteip": _client_ip(request),
    }

    try:
        response = httpx.post(
            SITEVERIFY_URL,
            data=payload,
            timeout=SITEVERIFY_TIMEOUT_SECONDS,
        )
    except httpx.RequestError:
        logger.exception(
            "Turnstile Siteverify network request failed."
        )

        raise _verification_unavailable() from None

    try:
        parsed_result = response.json()
    except ValueError:
        logger.error(
            "Turnstile Siteverify returned invalid JSON "
            "(HTTP %d).",
            response.status_code,
        )

        raise _verification_unavailable() from None

    if not isinstance(parsed_result, dict):
        logger.error(
            "Turnstile Siteverify returned an unexpected "
            "JSON type: %s.",
            type(parsed_result).__name__,
        )

        raise _verification_unavailable()

    result: dict[str, Any] = parsed_result
    error_codes = _error_codes(result)

    if (
        response.is_success
        and result.get("success") is True
    ):
        return

    configuration_failed = any(
        error_code in CONFIGURATION_ERROR_CODES
        for error_code in error_codes
    )

    upstream_failed = (
        response.status_code >= 500
        or any(
            error_code in TRANSIENT_ERROR_CODES
            for error_code in error_codes
        )
    )

    if configuration_failed or upstream_failed:
        logger.error(
            "Turnstile Siteverify unavailable or "
            "misconfigured: HTTP %d; error codes=%s.",
            response.status_code,
            error_codes,
        )

        raise _verification_unavailable()

    logger.warning(
        "Turnstile verification rejected: "
        "HTTP %d; error codes=%s.",
        response.status_code,
        error_codes,
    )

    raise _verification_rejected()
