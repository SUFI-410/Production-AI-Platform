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

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Human verification is temporarily "
                "unavailable."
            ),
        )

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

        response.raise_for_status()

        result: dict[str, Any] = response.json()

    except (
        httpx.HTTPError,
        ValueError,
    ):

        logger.exception(
            "Turnstile Siteverify request failed."
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Human verification failed.",
        )

    if result.get("success") is not True:
        error_codes = result.get(
            "error-codes",
            [],
        )

        logger.warning(
            "Turnstile verification rejected: %s",
            error_codes,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Human verification failed.",
        )
