"""Portal sessions use server-side tenant identity and safe Paddle responses."""

import json
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from api.dependencies import get_current_organization, get_db
from api.paddle_checkout_routes import get_portal_service, router
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rag.config import Config
from rag.database import Base
from rag.models import Organization, Plan, Subscription
from rag.paddle_portal import (
    PaddlePortalAPIError,
    PaddlePortalConfigurationError,
    PaddlePortalError,
    PaddlePortalService,
    PaddlePortalUnavailableError,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

CUSTOMER = "ctm_" + "a" * 26
SUBSCRIPTION = "sub_" + "b" * 26
URL = "https://sandbox-customer-portal.paddle.com/cpl_test?token=temporary"


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(Config, "PADDLE_ENVIRONMENT", "sandbox")
    monkeypatch.setattr(Config, "PADDLE_API_KEY", "test-key")


def add_subscription(db, organization_id):
    db.add(Organization(id=organization_id, name="Tenant"))
    plan = Plan(
        id=uuid4(),
        code="starter",
        name="Starter",
        invoice_checks_limit=250,
        users_limit=3,
        documents_limit=50,
    )
    db.add(plan)
    db.flush()
    subscription = Subscription(
        organization_id=organization_id,
        plan_id=plan.id,
        provider="paddle",
        provider_customer_id=CUSTOMER,
        provider_subscription_id=SUBSCRIPTION,
        status="canceled",
    )
    db.add(subscription)
    db.flush()
    return subscription


def payload(url=URL, customer=CUSTOMER):
    return {"data": {"customer_id": customer, "urls": {"general": {"overview": url}}}}


def test_portal_uses_tenant_customer_and_allows_canceled_subscription(db):
    org = uuid4()
    add_subscription(db, org)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == f"/customers/{CUSTOMER}/portal-sessions"
        assert request.url.host == "sandbox-api.paddle.com"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {"subscription_ids": [SUBSCRIPTION]}
        return httpx.Response(201, json=payload())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = PaddlePortalService(db, client=client)
        assert service.create_session(org) == URL
        assert service.create_session(org) == URL
    assert len(requests) == 2  # A fresh session on every click.


def test_another_tenant_cannot_get_existing_customer_portal(db):
    add_subscription(db, uuid4())
    client = Mock()
    with pytest.raises(PaddlePortalUnavailableError):
        PaddlePortalService(db, client=client).create_session(uuid4())
    client.post.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "other"),
        ("provider_customer_id", None),
        ("provider_subscription_id", None),
        ("provider_customer_id", "../customers/other"),
    ],
)
def test_unlinked_subscription_never_calls_paddle(db, field, value):
    org = uuid4()
    subscription = add_subscription(db, org)
    setattr(subscription, field, value)
    client = Mock()
    with pytest.raises(PaddlePortalUnavailableError):
        PaddlePortalService(db, client=client).create_session(org)
    client.post.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": None},
        payload(customer="wrong"),
        payload(url="https://evil.test/cpl_x"),
        payload(url="http://sandbox-customer-portal.paddle.com/cpl_x"),
        payload(url="https://sandbox-customer-portal.paddle.com@evil.test/cpl_x"),
        payload(url="https://sandbox-customer-portal.paddle.com/cpl_x\n"),
    ],
)
def test_invalid_paddle_response_is_rejected(db, body):
    org = uuid4()
    add_subscription(db, org)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(201, json=body))
    ) as client, pytest.raises(PaddlePortalAPIError):
        PaddlePortalService(db, client=client).create_session(org)


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
def test_paddle_http_errors_are_safe(db, status_code):
    org = uuid4()
    add_subscription(db, org)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status_code, text="private provider detail")
        )
    ) as client, pytest.raises(PaddlePortalAPIError, match="Unable to create"):
        PaddlePortalService(db, client=client).create_session(org)


def test_paddle_timeout_is_safe(db):
    org = uuid4()
    add_subscription(db, org)
    client = Mock()
    client.post.side_effect = httpx.ReadTimeout("private detail")
    with pytest.raises(PaddlePortalAPIError):
        PaddlePortalService(db, client=client).create_session(org)


@pytest.mark.parametrize(
    "setting,value", [("PADDLE_API_KEY", ""), ("PADDLE_ENVIRONMENT", "bad")]
)
def test_configuration_errors(db, monkeypatch, setting, value):
    org = uuid4()
    add_subscription(db, org)
    monkeypatch.setattr(Config, setting, value)
    client = Mock()
    with pytest.raises(PaddlePortalConfigurationError):
        PaddlePortalService(db, client=client).create_session(org)
    client.post.assert_not_called()


def make_app(service, org=None):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: Mock()
    app.dependency_overrides[get_portal_service] = lambda: service
    if org is not None:
        app.dependency_overrides[get_current_organization] = lambda: Organization(
            id=org
        )
    return app


def test_route_ignores_browser_customer_identity_and_disables_cache():
    org = uuid4()
    service = Mock()
    service.create_session.return_value = URL
    response = TestClient(make_app(service, org)).post(
        "/billing/portal",
        json={"organization_id": str(uuid4()), "customer_id": "other"},
    )
    assert response.status_code == 200
    assert response.json() == {"portal_url": URL}
    assert response.headers["cache-control"] == "no-store"
    service.create_session.assert_called_once_with(org)


def test_route_requires_authentication():
    service = Mock()
    response = TestClient(make_app(service)).post("/billing/portal")
    assert response.status_code == 401
    service.create_session.assert_not_called()


@pytest.mark.parametrize(
    "error,code",
    [
        (PaddlePortalUnavailableError, 404),
        (PaddlePortalConfigurationError, 503),
        (PaddlePortalAPIError, 502),
        (PaddlePortalError, 500),
    ],
)
def test_route_returns_safe_errors(error, code):
    service = Mock()
    service.create_session.side_effect = error("private detail")
    response = TestClient(make_app(service, uuid4())).post("/billing/portal")
    assert response.status_code == code
    assert "private detail" not in response.text


def test_live_environment(db, monkeypatch):
    org = uuid4()
    add_subscription(db, org)
    monkeypatch.setattr(Config, "PADDLE_ENVIRONMENT", "live")
    live_url = "https://customer-portal.paddle.com/cpl_test?token=temporary"

    def handler(request):
        assert request.url.host == "api.paddle.com"
        return httpx.Response(201, json=payload(url=live_url))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert PaddlePortalService(db, client=client).create_session(org) == live_url


def test_invalid_json(db):
    org = uuid4()
    add_subscription(db, org)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(201, text="not json")
        )
    ) as client, pytest.raises(PaddlePortalAPIError):
        PaddlePortalService(db, client=client).create_session(org)


def test_database_failure_never_calls_paddle():
    from sqlalchemy.exc import SQLAlchemyError

    db = Mock()
    db.scalar.side_effect = SQLAlchemyError("private detail")
    client = Mock()
    with pytest.raises(PaddlePortalError, match="Unable to resolve"):
        PaddlePortalService(db, client=client).create_session(uuid4())
    client.post.assert_not_called()
