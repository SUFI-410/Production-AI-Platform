"""Tests for authenticated Paddle checkout API routes."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import cast
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import api.paddle_checkout_routes as checkout_routes
from api.dependencies import get_current_organization
from api.main import app
from rag.billing_service import (
    BillingServiceError,
    BillingStatus,
    SubscriptionNotFoundError,
)
from rag.models import Organization
from rag.paddle_checkout import (
    PaddleCheckoutAPIError,
    PaddleCheckoutConfigurationError,
    PaddleCheckoutResult,
)


ORGANIZATION_ID = UUID(
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Keep FastAPI dependency overrides isolated between tests."""

    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


def _organization() -> Organization:
    """Return an authenticated test organization."""

    return Organization(
        id=ORGANIZATION_ID,
        name="Acme AI",
    )


def _override_organization(
    organization: Organization | None = None,
) -> None:
    """Override organization authentication for route tests."""

    resolved = (
        organization
        if organization is not None
        else _organization()
    )

    app.dependency_overrides[
        get_current_organization
    ] = lambda: cast(
        Organization,
        resolved,
    )


def _override_billing_service(service: Mock) -> None:
    app.dependency_overrides[
        checkout_routes.get_billing_service
    ] = lambda: service


def _billing_status() -> BillingStatus:
    period_start = datetime(2026, 9, 4, tzinfo=timezone.utc)
    period_end = datetime(2026, 10, 4, tzinfo=timezone.utc)

    return BillingStatus(
        organization_id=ORGANIZATION_ID,
        subscription_id=UUID(
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
        ),
        plan_code="starter",
        plan_name="Starter",
        subscription_status="active",
        access_mode="full",
        billing_interval="monthly",
        current_period_start=period_start,
        current_period_end=period_end,
        cancel_at_period_end=False,
        invoice_checks_used=2,
        invoice_checks_limit=250,
        invoice_checks_grace=10,
        can_run_invoice_check=True,
        usage_period_start=period_start,
        usage_period_end=period_end,
        documents_used=12,
        documents_limit=50,
        can_upload_document=True,
        users_used=1,
        users_limit=3,
        api_access=False,
        audit_logs=False,
    )


def test_billing_status_returns_authenticated_organization_snapshot() -> None:
    _override_organization()
    service = Mock()
    service.status.return_value = _billing_status()
    _override_billing_service(service)

    response = TestClient(app).get("/billing/status")

    assert response.status_code == 200
    assert response.json()["plan_code"] == "starter"
    assert response.json()["subscription_status"] == "active"
    assert response.json()["invoice_checks_used"] == 2
    assert response.json()["documents_used"] == 12
    assert response.json()["users_used"] == 1
    service.status.assert_called_once_with(ORGANIZATION_ID)


def test_billing_status_returns_404_without_subscription() -> None:
    _override_organization()
    service = Mock()
    service.status.side_effect = SubscriptionNotFoundError(
        "Organization has no subscription."
    )
    _override_billing_service(service)

    response = TestClient(app).get("/billing/status")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Organization has no subscription."
    }


def test_billing_status_hides_internal_billing_errors() -> None:
    _override_organization()
    service = Mock()
    service.status.side_effect = BillingServiceError(
        "Sensitive internal state."
    )
    _override_billing_service(service)

    response = TestClient(app).get("/billing/status")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Unable to resolve billing status."
    }


def test_billing_status_requires_authentication() -> None:
    response = TestClient(app).get("/billing/status")

    assert response.status_code == 401


def test_checkout_creates_transaction_for_authenticated_organization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route must use the authenticated organization ID."""

    _override_organization()

    create_checkout = Mock(
        return_value=PaddleCheckoutResult(
            transaction_id="txn_123",
            checkout_url=(
                "https://checkout.example.com/"
                "?_ptxn=txn_123"
            ),
        )
    )

    service_instance = Mock()
    service_instance.create_checkout = create_checkout

    service_factory = Mock(
        return_value=service_instance
    )

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        service_factory,
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "professional",
            "billing_interval": "annual",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "transaction_id": "txn_123",
        "checkout_url": (
            "https://checkout.example.com/"
            "?_ptxn=txn_123"
        ),
    }

    service_factory.assert_called_once_with()

    create_checkout.assert_called_once_with(
        organization_id=ORGANIZATION_ID,
        plan_code="professional",
        billing_interval="annual",
    )


def test_request_cannot_supply_organization_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Organization identity must never be accepted from the client body.
    """

    _override_organization()

    service_factory = Mock()

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        service_factory,
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "starter",
            "billing_interval": "monthly",
            "organization_id": (
                "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
            ),
        },
    )

    assert response.status_code == 422

    service_factory.assert_not_called()


@pytest.mark.parametrize(
    "plan_code",
    [
        "trial",
        "enterprise",
        "invalid",
        "",
    ],
)
def test_invalid_plan_is_rejected(
    plan_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the three self-service paid plans are accepted."""

    _override_organization()

    service_factory = Mock()

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        service_factory,
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": plan_code,
            "billing_interval": "monthly",
        },
    )

    assert response.status_code == 422

    service_factory.assert_not_called()


@pytest.mark.parametrize(
    "billing_interval",
    [
        "weekly",
        "quarterly",
        "yearly",
        "",
    ],
)
def test_invalid_billing_interval_is_rejected(
    billing_interval: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only monthly and annual billing are accepted."""

    _override_organization()

    service_factory = Mock()

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        service_factory,
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "starter",
            "billing_interval": billing_interval,
        },
    )

    assert response.status_code == 422

    service_factory.assert_not_called()


def test_missing_plan_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan selection is required."""

    _override_organization()

    service_factory = Mock()

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        service_factory,
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "billing_interval": "monthly",
        },
    )

    assert response.status_code == 422

    service_factory.assert_not_called()


def test_missing_billing_interval_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A billing interval is required."""

    _override_organization()

    service_factory = Mock()

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        service_factory,
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "starter",
        },
    )

    assert response.status_code == 422

    service_factory.assert_not_called()


def test_paddle_configuration_error_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing server-side Paddle configuration must fail safely."""

    _override_organization()

    create_checkout = Mock(
        side_effect=PaddleCheckoutConfigurationError(
            "Paddle API key is not configured."
        )
    )

    service_instance = Mock()
    service_instance.create_checkout = create_checkout

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        Mock(
            return_value=service_instance
        ),
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "starter",
            "billing_interval": "monthly",
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Billing checkout is not configured."
    }


def test_paddle_api_error_returns_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paddle API failures must not expose provider internals."""

    _override_organization()

    create_checkout = Mock(
        side_effect=PaddleCheckoutAPIError(
            "Paddle rejected checkout creation."
        )
    )

    service_instance = Mock()
    service_instance.create_checkout = create_checkout

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        Mock(
            return_value=service_instance
        ),
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "business",
            "billing_interval": "annual",
        },
    )

    assert response.status_code == 502

    assert response.json() == {
        "detail": "Unable to create billing checkout."
    }


def test_checkout_endpoint_requires_authentication() -> None:
    """
    Without an organization dependency override, normal authentication
    must reject an unauthenticated request.
    """

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "starter",
            "billing_interval": "monthly",
        },
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": (
            "Invalid or missing authentication credentials."
        )
    }


def test_checkout_response_does_not_expose_organization_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tenant identity must not appear in the checkout API response."""

    _override_organization()

    result = PaddleCheckoutResult(
        transaction_id="txn_456",
        checkout_url="https://checkout.example.com/txn_456",
    )

    service_instance = Mock()
    service_instance.create_checkout = Mock(
        return_value=result
    )

    monkeypatch.setattr(
        checkout_routes,
        "PaddleCheckoutService",
        Mock(
            return_value=service_instance
        ),
    )

    client = TestClient(app)

    response = client.post(
        "/billing/checkout",
        json={
            "plan_code": "starter",
            "billing_interval": "monthly",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body == {
        "transaction_id": "txn_456",
        "checkout_url": (
            "https://checkout.example.com/txn_456"
        ),
    }

    assert "organization_id" not in body
