"""
Create Paddle checkout transactions for authenticated organizations.

This module resolves local plan selections to configured Paddle price IDs
and creates transactions using Paddle's server-side API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx

from rag.config import Config
from rag.models import BillingInterval, PlanCode


class PaddleCheckoutError(RuntimeError):
    """Base error for Paddle checkout creation."""


class PaddleCheckoutConfigurationError(
    PaddleCheckoutError
):
    """Raised when Paddle checkout configuration is invalid."""


class PaddleCheckoutAPIError(
    PaddleCheckoutError
):
    """Raised when Paddle rejects or fails a checkout request."""


@dataclass(frozen=True)
class PaddleCheckoutResult:
    """Checkout transaction returned to the API layer."""

    transaction_id: str
    checkout_url: str


def _api_base_url() -> str:
    """Return the Paddle API base URL for the configured environment."""

    environment = Config.PADDLE_ENVIRONMENT.strip().lower()

    if environment == "sandbox":
        return "https://sandbox-api.paddle.com"

    if environment == "live":
        return "https://api.paddle.com"

    raise PaddleCheckoutConfigurationError(
        "Unsupported Paddle environment."
    )


def _price_id(
    *,
    plan_code: str,
    billing_interval: str,
) -> str:
    """Resolve a local plan and interval to its Paddle price ID."""

    mappings = {
        (
            PlanCode.STARTER.value,
            BillingInterval.MONTHLY.value,
        ): Config.PADDLE_STARTER_MONTHLY_PRICE_ID,
        (
            PlanCode.STARTER.value,
            BillingInterval.ANNUAL.value,
        ): Config.PADDLE_STARTER_ANNUAL_PRICE_ID,
        (
            PlanCode.PROFESSIONAL.value,
            BillingInterval.MONTHLY.value,
        ): Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID,
        (
            PlanCode.PROFESSIONAL.value,
            BillingInterval.ANNUAL.value,
        ): Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID,
        (
            PlanCode.BUSINESS.value,
            BillingInterval.MONTHLY.value,
        ): Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID,
        (
            PlanCode.BUSINESS.value,
            BillingInterval.ANNUAL.value,
        ): Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID,
    }

    price_id = mappings.get(
        (
            plan_code,
            billing_interval,
        )
    )

    if price_id is None:
        raise PaddleCheckoutConfigurationError(
            "Unsupported Paddle plan or billing interval."
        )

    price_id = price_id.strip()

    if not price_id:
        raise PaddleCheckoutConfigurationError(
            "Paddle price ID is not configured."
        )

    return price_id


def _api_key() -> str:
    """Return the configured Paddle API key."""

    api_key = Config.PADDLE_API_KEY.strip()

    if not api_key:
        raise PaddleCheckoutConfigurationError(
            "Paddle API key is not configured."
        )

    return api_key


def _response_data(
    response: httpx.Response,
) -> dict[str, Any]:
    """Validate the top-level Paddle API response body."""

    try:
        payload = response.json()
    except ValueError as exc:
        raise PaddleCheckoutAPIError(
            "Paddle returned an invalid response."
        ) from exc

    if not isinstance(payload, dict):
        raise PaddleCheckoutAPIError(
            "Paddle returned an invalid response."
        )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise PaddleCheckoutAPIError(
            "Paddle response is missing transaction data."
        )

    return data


def _checkout_result(
    data: dict[str, Any],
) -> PaddleCheckoutResult:
    """Extract the transaction ID and Paddle checkout URL."""

    transaction_id = data.get("id")

    checkout = data.get("checkout")

    if (
        not isinstance(transaction_id, str)
        or not transaction_id.strip()
        or not isinstance(checkout, dict)
    ):
        raise PaddleCheckoutAPIError(
            "Paddle response is missing checkout details."
        )

    checkout_url = checkout.get("url")

    if (
        not isinstance(checkout_url, str)
        or not checkout_url.strip()
    ):
        raise PaddleCheckoutAPIError(
            "Paddle response is missing checkout URL."
        )

    return PaddleCheckoutResult(
        transaction_id=transaction_id.strip(),
        checkout_url=checkout_url.strip(),
    )


class PaddleCheckoutService:
    """Create Paddle transactions that can be opened in Checkout."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client

    def create_checkout(
        self,
        *,
        organization_id: UUID,
        plan_code: str,
        billing_interval: str,
    ) -> PaddleCheckoutResult:
        """Create one Paddle checkout transaction."""

        price_id = _price_id(
            plan_code=plan_code,
            billing_interval=billing_interval,
        )

        request_payload = {
            "items": [
                {
                    "price_id": price_id,
                    "quantity": 1,
                }
            ],
            "collection_mode": "automatic",
            "custom_data": {
                "organization_id": str(
                    organization_id
                ),
            },
        }

        headers = {
            "Authorization": (
                f"Bearer {_api_key()}"
            ),
            "Content-Type": "application/json",
            "Paddle-Version": "1",
        }

        owns_client = self._client is None

        client = (
            self._client
            if self._client is not None
            else httpx.Client(
                timeout=Config.REQUEST_TIMEOUT
            )
        )

        try:
            response = client.post(
                f"{_api_base_url()}/transactions",
                headers=headers,
                json=request_payload,
            )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise PaddleCheckoutAPIError(
                    "Paddle rejected checkout creation."
                ) from exc

            data = _response_data(
                response
            )

            return _checkout_result(
                data
            )

        except httpx.RequestError as exc:
            raise PaddleCheckoutAPIError(
                "Unable to reach Paddle."
            ) from exc

        finally:
            if owns_client:
                client.close()
