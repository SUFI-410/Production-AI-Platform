"""Tests for subscription entitlement and usage enforcement."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag.billing_service import (
    BillingService,
    DocumentUploadNotAllowedError,
    InvoiceCheckNotAllowedError,
    SubscriptionNotFoundError,
)
from rag.database import Base
from rag.models import (
    Document,
    DocumentStatus,
    DocumentType,
    Organization,
    Plan,
    Subscription,
    SubscriptionStatus,
    User,
)


@pytest.fixture
def session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite://")

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def _now() -> datetime:
    return datetime(
        2026,
        9,
        2,
        12,
        0,
        tzinfo=timezone.utc,
    )


def _create_organization(
    session: Session,
) -> Organization:
    organization = Organization(
        id=uuid4(),
        name="Test Organization",
    )

    session.add(organization)
    session.flush()

    return organization


def _create_plan(
    session: Session,
    *,
    code: str = "starter",
    limit: int = 3,
    grace: int = 1,
    documents_limit: int = 50,
) -> Plan:
    plan = Plan(
        id=uuid4(),
        code=code,
        name=code.title(),
        monthly_price_cents=14900,
        annual_price_cents=149000,
        invoice_checks_limit=limit,
        invoice_checks_grace=grace,
        users_limit=3,
        documents_limit=documents_limit,
        api_access=False,
        audit_logs=False,
        is_active=True,
    )

    session.add(plan)
    session.flush()

    return plan


def _create_subscription(
    session: Session,
    *,
    organization: Organization,
    plan: Plan,
    status: str,
    now: datetime,
    trial: bool = False,
) -> Subscription:
    if trial:
        subscription = Subscription(
            id=uuid4(),
            organization_id=organization.id,
            plan_id=plan.id,
            status=status,
            trial_started_at=now - timedelta(days=1),
            trial_ends_at=now + timedelta(days=13),
        )
    else:
        subscription = Subscription(
            id=uuid4(),
            organization_id=organization.id,
            plan_id=plan.id,
            status=status,
            billing_interval="monthly",
            current_period_start=now - timedelta(days=5),
            current_period_end=now + timedelta(days=25),
        )

    session.add(subscription)
    session.flush()

    return subscription


def _create_document(
    session: Session,
    *,
    organization: Organization,
) -> Document:
    document = Document(
        id=uuid4(),
        organization_id=organization.id,
        uploaded_by_user_id=None,
        original_filename="requirement.md",
        content_type="text/markdown",
        size_bytes=1,
        storage_key=f"{organization.id}/{uuid4()}.md",
        document_type=DocumentType.CONTRACT.value,
        status=DocumentStatus.UPLOADED.value,
    )

    session.add(document)
    session.flush()

    return document


def test_missing_subscription_is_rejected(
    session: Session,
) -> None:
    organization = _create_organization(session)

    service = BillingService(session)

    with pytest.raises(SubscriptionNotFoundError):
        service.entitlement(
            organization.id,
            now=_now(),
        )


def test_active_subscription_has_full_access(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(session)

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.ACTIVE.value,
        now=now,
    )

    entitlement = BillingService(session).entitlement(
        organization.id,
        now=now,
    )

    assert entitlement.access_mode == "full"
    assert entitlement.invoice_checks_used == 0
    assert entitlement.invoice_checks_limit == 3
    assert entitlement.invoice_checks_grace == 1
    assert entitlement.can_run_invoice_check is True
    assert entitlement.in_grace_buffer is False


def test_active_trial_has_full_access(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)

    plan = _create_plan(
        session,
        code="trial",
        limit=25,
        grace=0,
    )

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.TRIALING.value,
        now=now,
        trial=True,
    )

    entitlement = BillingService(session).entitlement(
        organization.id,
        now=now,
    )

    assert entitlement.access_mode == "full"
    assert entitlement.plan_code == "trial"
    assert entitlement.invoice_checks_limit == 25
    assert entitlement.can_run_invoice_check is True


def test_expired_trial_is_read_only(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)

    plan = _create_plan(
        session,
        code="trial",
        limit=25,
        grace=0,
    )

    subscription = Subscription(
        id=uuid4(),
        organization_id=organization.id,
        plan_id=plan.id,
        status=SubscriptionStatus.TRIALING.value,
        trial_started_at=now - timedelta(days=15),
        trial_ends_at=now - timedelta(days=1),
        read_only_until=now + timedelta(days=13),
    )

    session.add(subscription)
    session.flush()

    entitlement = BillingService(session).entitlement(
        organization.id,
        now=now,
    )

    assert entitlement.access_mode == "read_only"
    assert entitlement.can_run_invoice_check is False


def test_past_due_inside_grace_keeps_full_access(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(session)

    subscription = _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.PAST_DUE.value,
        now=now,
    )

    subscription.past_due_since = now - timedelta(days=2)
    subscription.grace_period_ends_at = now + timedelta(days=5)

    entitlement = BillingService(session).entitlement(
        organization.id,
        now=now,
    )

    assert entitlement.access_mode == "full"
    assert entitlement.can_run_invoice_check is True


def test_past_due_after_grace_is_read_only(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(session)

    subscription = _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.PAST_DUE.value,
        now=now,
    )

    subscription.past_due_since = now - timedelta(days=8)
    subscription.grace_period_ends_at = now - timedelta(days=1)

    entitlement = BillingService(session).entitlement(
        organization.id,
        now=now,
    )

    assert entitlement.access_mode == "read_only"
    assert entitlement.can_run_invoice_check is False


def test_canceled_subscription_is_read_only(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(session)

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.CANCELED.value,
        now=now,
    )

    entitlement = BillingService(session).entitlement(
        organization.id,
        now=now,
    )

    assert entitlement.access_mode == "read_only"
    assert entitlement.can_run_invoice_check is False


def test_invoice_usage_enters_grace_buffer(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)

    plan = _create_plan(
        session,
        limit=2,
        grace=1,
    )

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.ACTIVE.value,
        now=now,
    )

    service = BillingService(session)

    first = service.consume_invoice_check(
        organization.id,
        now=now,
    )

    second = service.consume_invoice_check(
        organization.id,
        now=now,
    )

    assert first.invoice_checks_used == 1
    assert first.in_grace_buffer is False

    assert second.invoice_checks_used == 2
    assert second.in_grace_buffer is True
    assert second.can_run_invoice_check is True


def test_invoice_usage_hard_blocks_after_grace(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)

    plan = _create_plan(
        session,
        limit=2,
        grace=1,
    )

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.ACTIVE.value,
        now=now,
    )

    service = BillingService(session)

    service.consume_invoice_check(
        organization.id,
        now=now,
    )

    service.consume_invoice_check(
        organization.id,
        now=now,
    )

    third = service.consume_invoice_check(
        organization.id,
        now=now,
    )

    assert third.invoice_checks_used == 3
    assert third.can_run_invoice_check is False
    assert third.in_grace_buffer is False

    with pytest.raises(InvoiceCheckNotAllowedError):
        service.consume_invoice_check(
            organization.id,
            now=now,
        )


def test_read_only_subscription_cannot_consume_usage(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(session)

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.READ_ONLY.value,
        now=now,
    )

    service = BillingService(session)

    with pytest.raises(InvoiceCheckNotAllowedError):
        service.consume_invoice_check(
            organization.id,
            now=now,
        )


def test_expired_trial_cannot_consume_usage(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)

    plan = _create_plan(
        session,
        code="trial",
        limit=25,
        grace=0,
    )

    subscription = Subscription(
        id=uuid4(),
        organization_id=organization.id,
        plan_id=plan.id,
        status=SubscriptionStatus.TRIALING.value,
        trial_started_at=now - timedelta(days=15),
        trial_ends_at=now - timedelta(days=1),
        read_only_until=now + timedelta(days=13),
    )

    session.add(subscription)
    session.flush()

    service = BillingService(session)

    with pytest.raises(InvoiceCheckNotAllowedError):
        service.consume_invoice_check(
            organization.id,
            now=now,
        )


def test_annual_subscription_still_gets_monthly_usage_period(
    session: Session,
) -> None:
    now = datetime(
        2026,
        11,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    organization = _create_organization(session)
    plan = _create_plan(session)

    subscription = Subscription(
        id=uuid4(),
        organization_id=organization.id,
        plan_id=plan.id,
        status=SubscriptionStatus.ACTIVE.value,
        billing_interval="annual",
        current_period_start=datetime(
            2026,
            9,
            12,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        current_period_end=datetime(
            2027,
            9,
            12,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    session.add(subscription)
    session.flush()

    entitlement = BillingService(session).entitlement(
        organization.id,
        now=now,
    )

    assert entitlement.period_start == datetime(
        2026,
        11,
        12,
        12,
        0,
        tzinfo=timezone.utc,
    )

    assert entitlement.period_end == datetime(
        2026,
        12,
        12,
        12,
        0,
        tzinfo=timezone.utc,
    )

    assert entitlement.invoice_checks_used == 0


def test_monthly_usage_resets_into_new_period(
    session: Session,
) -> None:
    organization = _create_organization(session)
    plan = _create_plan(session)

    subscription = Subscription(
        id=uuid4(),
        organization_id=organization.id,
        plan_id=plan.id,
        status=SubscriptionStatus.ACTIVE.value,
        billing_interval="monthly",
        current_period_start=datetime(
            2026,
            9,
            12,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        current_period_end=datetime(
            2026,
            12,
            12,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    session.add(subscription)
    session.flush()

    service = BillingService(session)

    september = datetime(
        2026,
        9,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    october = datetime(
        2026,
        10,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    first_period = service.consume_invoice_check(
        organization.id,
        now=september,
    )

    second_period = service.entitlement(
        organization.id,
        now=october,
    )

    assert first_period.invoice_checks_used == 1

    assert first_period.period_start == datetime(
        2026,
        9,
        12,
        12,
        0,
        tzinfo=timezone.utc,
    )

    assert second_period.invoice_checks_used == 0

    assert second_period.period_start == datetime(
        2026,
        10,
        12,
        12,
        0,
        tzinfo=timezone.utc,
    )


def test_document_upload_is_allowed_below_plan_limit(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(
        session,
        documents_limit=2,
    )

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.ACTIVE.value,
        now=now,
    )

    _create_document(
        session,
        organization=organization,
    )

    BillingService(session).ensure_document_upload_allowed(
        organization.id,
        now=now,
    )


def test_document_upload_is_rejected_at_plan_limit(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(
        session,
        documents_limit=1,
    )

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.ACTIVE.value,
        now=now,
    )

    _create_document(
        session,
        organization=organization,
    )

    with pytest.raises(
        DocumentUploadNotAllowedError,
        match="Document allowance has been exhausted.",
    ):
        BillingService(session).ensure_document_upload_allowed(
            organization.id,
            now=now,
        )


def test_document_upload_is_rejected_for_read_only_subscription(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(session)

    _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.CANCELED.value,
        now=now,
    )

    with pytest.raises(
        DocumentUploadNotAllowedError,
        match="Subscription is read-only.",
    ):
        BillingService(session).ensure_document_upload_allowed(
            organization.id,
            now=now,
        )


def test_billing_status_returns_read_only_capacity_snapshot(
    session: Session,
) -> None:
    now = _now()
    organization = _create_organization(session)
    plan = _create_plan(session, documents_limit=2)
    subscription = _create_subscription(
        session,
        organization=organization,
        plan=plan,
        status=SubscriptionStatus.ACTIVE.value,
        now=now,
    )
    session.add(
        User(
            id=uuid4(),
            organization_id=organization.id,
            email="owner@example.com",
            password_hash="stored-password-hash",
            is_active=True,
        )
    )
    _create_document(session, organization=organization)
    session.flush()

    result = BillingService(session).status(
        organization.id,
        now=now,
    )

    assert result.subscription_id == subscription.id
    assert result.plan_code == "starter"
    assert result.plan_name == "Starter"
    assert result.subscription_status == "active"
    assert result.access_mode == "full"
    assert result.billing_interval == "monthly"
    assert result.invoice_checks_used == 0
    assert result.invoice_checks_limit == 3
    assert result.invoice_checks_grace == 1
    assert result.can_run_invoice_check is True
    assert result.documents_used == 1
    assert result.documents_limit == 2
    assert result.can_upload_document is True
    assert result.users_used == 1
    assert result.users_limit == 3
    assert result.api_access is False
    assert result.audit_logs is False
    assert list(session.new) == []
