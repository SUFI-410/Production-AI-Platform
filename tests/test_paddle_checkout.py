"""Tests for Paddle checkout transaction creation."""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest

from rag.config import Config
from rag.models import BillingInterval, PlanCode
from rag.paddle_checkout import (
    PaddleCheckoutAPIError,
    PaddleCheckoutConfigurationError,
    PaddleCheckoutService,
)


ORGANIZATION_ID = UUID(
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
)


@pytest.fixture(autouse=True)
def restore_paddle_config():
    original_environment = Config.PADDLE_ENVIRONMENT
    original_api_key = Config.PADDLE_API_KEY

    original_starter_monthly = (
        Config.PADDLE_STARTER_MONTHLY_PRICE_ID
    )

    original_starter_annual = (
        Config.PADDLE_STARTER_ANNUAL_PRICE_ID
    )

    original_professional_monthly = (
        Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID
    )

    original_professional_annual = (
        Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID
    )

    original_business_monthly = (
        Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID
    )

    original_business_annual = (
        Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID
    )

    Config.PADDLE_ENVIRONMENT = "sandbox"
    Config.PADDLE_API_KEY = "test-api-key"

    Config.PADDLE_STARTER_MONTHLY_PRICE_ID = (
        "pri_starter_monthly"
    )

    Config.PADDLE_STARTER_ANNUAL_PRICE_ID = (
        "pri_starter_annual"
    )

    Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID = (
        "pri_professional_monthly"
    )

    Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID = (
        "pri_professional_annual"
    )

    Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID = (
        "pri_business_monthly"
    )

    Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID = (
        "pri_business_annual"
    )

    yield

    Config.PADDLE_ENVIRONMENT = original_environment
    Config.PADDLE_API_KEY = original_api_key

    Config.PADDLE_STARTER_MONTHLY_PRICE_ID = (
        original_starter_monthly
    )

    Config.PADDLE_STARTER_ANNUAL_PRICE_ID = (
        original_starter_annual
    )

    Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID = (
        original_professional_monthly
    )

    Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID = (
        original_professional_annual
    )

    Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID = (
        original_business_monthly
    )

    Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID = (
        original_business_annual
    )


@pytest.mark.parametrize(
    (
        "plan_code",
        "billing_interval",
        "expected_price_id",
    ),
    [
        (
            PlanCode.STARTER.value,
            BillingInterval.MONTHLY.value,
            "pri_starter_monthly",
        ),
        (
            PlanCode.STARTER.value,
            BillingInterval.ANNUAL.value,
            "pri_starter_annual",
        ),
        (
            PlanCode.PROFESSIONAL.value,
            BillingInterval.MONTHLY.value,
            "pri_professional_monthly",
        ),
        (
            PlanCode.PROFESSIONAL.value,
            BillingInterval.ANNUAL.value,
            "pri_professional_annual",
        ),
        (
            PlanCode.BUSINESS.value,
            BillingInterval.MONTHLY.value,
            "pri_business_monthly",
        ),
        (
            PlanCode.BUSINESS.value,
            BillingInterval.ANNUAL.value,
            "pri_business_annual",
        ),
    ],
)
def test_creates_checkout_for_each_configured_plan(
    plan_code: str,
    billing_interval: str,
    expected_price_id: str,
) -> None:
    captured_request: httpx.Request | None = None

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal captured_request

        captured_request = request

        return httpx.Response(
            201,
            json={
                "data": {
                    "id": "txn_123",
                    "checkout": {
                        "url": (
                            "https://checkout.example.com/"
                            "?_ptxn=txn_123"
                        )
                    },
                }
            },
        )

    transport = httpx.MockTransport(
        handler
    )

    with httpx.Client(
        transport=transport
    ) as client:
        service = PaddleCheckoutService(
            client=client
        )

        result = service.create_checkout(
            organization_id=ORGANIZATION_ID,
            plan_code=plan_code,
            billing_interval=billing_interval,
        )

    assert result.transaction_id == "txn_123"

    assert result.checkout_url == (
        "https://checkout.example.com/"
        "?_ptxn=txn_123"
    )

    assert captured_request is not None

    assert str(captured_request.url) == (
        "https://sandbox-api.paddle.com/transactions"
    )

    assert (
        captured_request.headers["authorization"]
        == "Bearer test-api-key"
    )

    assert (
        captured_request.headers["paddle-version"]
        == "1"
    )

    request_payload = (
        captured_request.read().decode("utf-8")
    )

    assert f'"price_id":"{expected_price_id}"' in request_payload

    assert '"quantity":1' in request_payload

    assert (
        f'"organization_id":"{ORGANIZATION_ID}"'
        in request_payload
    )


def test_live_environment_uses_live_api() -> None:
    Config.PADDLE_ENVIRONMENT = "live"

    captured_url = ""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal captured_url

        captured_url = str(
            request.url
        )

        return httpx.Response(
            201,
            json={
                "data": {
                    "id": "txn_live",
                    "checkout": {
                        "url": "https://checkout.example.com/live"
                    },
                }
            },
        )

    transport = httpx.MockTransport(
        handler
    )

    with httpx.Client(
        transport=transport
    ) as client:
        service = PaddleCheckoutService(
            client=client
        )

        service.create_checkout(
            organization_id=ORGANIZATION_ID,
            plan_code=PlanCode.STARTER.value,
            billing_interval=BillingInterval.MONTHLY.value,
        )

    assert captured_url == (
        "https://api.paddle.com/transactions"
    )


def test_missing_api_key_is_rejected() -> None:
    Config.PADDLE_API_KEY = ""

    service = PaddleCheckoutService()

    with pytest.raises(
        PaddleCheckoutConfigurationError,
        match="API key",
    ):
        service.create_checkout(
            organization_id=ORGANIZATION_ID,
            plan_code=PlanCode.STARTER.value,
            billing_interval=BillingInterval.MONTHLY.value,
        )


def test_missing_price_id_is_rejected() -> None:
    Config.PADDLE_STARTER_MONTHLY_PRICE_ID = ""

    service = PaddleCheckoutService()

    with pytest.raises(
        PaddleCheckoutConfigurationError,
        match="price ID",
    ):
        service.create_checkout(
            organization_id=ORGANIZATION_ID,
            plan_code=PlanCode.STARTER.value,
            billing_interval=BillingInterval.MONTHLY.value,
        )


def test_unsupported_plan_is_rejected() -> None:
    service = PaddleCheckoutService()

    with pytest.raises(
        PaddleCheckoutConfigurationError,
        match="Unsupported Paddle plan",
    ):
        service.create_checkout(
            organization_id=ORGANIZATION_ID,
            plan_code="enterprise",
            billing_interval=BillingInterval.MONTHLY.value,
        )


def test_invalid_environment_is_rejected() -> None:
    Config.PADDLE_ENVIRONMENT = "invalid"

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            201,
            json={},
        )
    )

    with httpx.Client(
        transport=transport
    ) as client:
        service = PaddleCheckoutService(
            client=client
        )

        with pytest.raises(
            PaddleCheckoutConfigurationError,
            match="environment",
        ):
            service.create_checkout(
                organization_id=ORGANIZATION_ID,
                plan_code=PlanCode.STARTER.value,
                billing_interval=BillingInterval.MONTHLY.value,
            )


def test_paddle_http_error_is_wrapped() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "type": "request_error"
                }
            },
        )
    )

    with httpx.Client(
        transport=transport
    ) as client:
        service = PaddleCheckoutService(
            client=client
        )

        with pytest.raises(
            PaddleCheckoutAPIError,
            match="rejected",
        ):
            service.create_checkout(
                organization_id=ORGANIZATION_ID,
                plan_code=PlanCode.STARTER.value,
                billing_interval=BillingInterval.MONTHLY.value,
            )


def test_invalid_json_response_is_rejected() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            201,
            content=b"not-json",
        )
    )

    with httpx.Client(
        transport=transport
    ) as client:
        service = PaddleCheckoutService(
            client=client
        )

        with pytest.raises(
            PaddleCheckoutAPIError,
            match="invalid response",
        ):
            service.create_checkout(
                organization_id=ORGANIZATION_ID,
                plan_code=PlanCode.STARTER.value,
                billing_interval=BillingInterval.MONTHLY.value,
            )


def test_missing_checkout_url_is_rejected() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            201,
            json={
                "data": {
                    "id": "txn_123",
                    "checkout": {
                        "url": None
                    },
                }
            },
        )
    )

    with httpx.Client(
        transport=transport
    ) as client:
        service = PaddleCheckoutService(
            client=client
        )

        with pytest.raises(
            PaddleCheckoutAPIError,
            match="checkout URL",
        ):
            service.create_checkout(
                organization_id=ORGANIZATION_ID,
                plan_code=PlanCode.STARTER.value,
                billing_interval=BillingInterval.MONTHLY.value,
            )
