"""Tests for Paddle subscription persistence."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag.database import Base
from rag.models import (
    BillingEvent,
    BillingInterval,
    Organization,
    Plan,
    PlanCode,
    Subscription,
    SubscriptionStatus,
)
from rag.paddle_subscription_service import (
    PAST_DUE_GRACE_DAYS,
    PaddleOrganizationNotFoundError,
    PaddlePlanNotFoundError,
    PaddleSubscriptionConflictError,
    PaddleSubscriptionService,
)
from rag.paddle_subscription_sync import PaddleSubscriptionState


ORGANIZATION_ID = UUID(
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
)

OTHER_ORGANIZATION_ID = UUID(
    "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
)

STARTER_PLAN_ID = UUID(
    "a2222222-2222-4222-8222-222222222222"
)

PROFESSIONAL_PLAN_ID = UUID(
    "b3333333-3333-4333-8333-333333333333"
)

EVENT_TIME = datetime(
    2026,
    9,
    2,
    8,
    0,
    tzinfo=timezone.utc,
)

PERIOD_START = datetime(
    2026,
    9,
    1,
    tzinfo=timezone.utc,
)

PERIOD_END = datetime(
    2026,
    10,
    1,
    tzinfo=timezone.utc,
)


@pytest.fixture()
def db() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )

    Base.metadata.create_all(
        engine
    )

    factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    session = factory()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _organization(
    *,
    organization_id: UUID = ORGANIZATION_ID,
) -> Organization:
    return Organization(
        id=organization_id,
        name="Acme AI",
    )


def _plan(
    *,
    plan_id: UUID = STARTER_PLAN_ID,
    code: str = PlanCode.STARTER.value,
    active: bool = True,
) -> Plan:
    return Plan(
        id=plan_id,
        code=code,
        name=code.title(),
        monthly_price_cents=14900,
        annual_price_cents=149000,
        invoice_checks_limit=250,
        invoice_checks_grace=25,
        users_limit=3,
        documents_limit=50,
        api_access=False,
        audit_logs=False,
        is_active=active,
    )


def _state(
    *,
    organization_id: UUID = ORGANIZATION_ID,
    provider_customer_id: str = "ctm_123",
    provider_subscription_id: str = "sub_123",
    plan_code: str = PlanCode.STARTER.value,
    billing_interval: str = BillingInterval.MONTHLY.value,
    status: str = SubscriptionStatus.ACTIVE.value,
    current_period_start: datetime | None = PERIOD_START,
    current_period_end: datetime | None = PERIOD_END,
    cancel_at_period_end: bool = False,
    event_occurred_at: datetime = EVENT_TIME,
) -> PaddleSubscriptionState:
    return PaddleSubscriptionState(
        organization_id=organization_id,
        provider_customer_id=provider_customer_id,
        provider_subscription_id=provider_subscription_id,
        plan_code=plan_code,
        billing_interval=billing_interval,
        status=status,
        current_period_start=current_period_start,
        current_period_end=current_period_end,
        cancel_at_period_end=cancel_at_period_end,
        event_occurred_at=event_occurred_at,
    )


def _billing_event(
    *,
    event_id: str = "evt_123",
) -> BillingEvent:
    return BillingEvent(
        id=uuid4(),
        provider="paddle",
        provider_event_id=event_id,
        event_type="subscription.updated",
        payload={
            "event_id": event_id,
        },
    )


def _existing_subscription(
    *,
    organization_id: UUID = ORGANIZATION_ID,
    plan_id: UUID = STARTER_PLAN_ID,
    provider_subscription_id: str = "sub_123",
    provider_customer_id: str = "ctm_123",
    status: str = SubscriptionStatus.ACTIVE.value,
    provider_updated_at: datetime | None = None,
) -> Subscription:
    return Subscription(
        organization_id=organization_id,
        plan_id=plan_id,
        scheduled_plan_id=None,
        provider="paddle",
        provider_customer_id=provider_customer_id,
        provider_subscription_id=provider_subscription_id,
        provider_updated_at=provider_updated_at,
        status=status,
        billing_interval=BillingInterval.MONTHLY.value,
        current_period_start=PERIOD_START,
        current_period_end=PERIOD_END,
        cancel_at_period_end=False,
        trial_started_at=None,
        trial_ends_at=None,
        read_only_until=None,
        past_due_since=None,
        grace_period_ends_at=None,
    )


def _seed_base(
    db: Session,
) -> None:
    db.add(
        _organization()
    )

    db.add(
        _plan()
    )

    db.flush()


def test_creates_new_active_subscription(
    db: Session,
) -> None:
    _seed_base(db)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    result = service.synchronize(
        state=_state(),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert result.applied is True
    assert result.stale is False

    subscription = result.subscription

    assert subscription.organization_id == ORGANIZATION_ID
    assert subscription.plan_id == STARTER_PLAN_ID

    assert subscription.provider == "paddle"
    assert subscription.provider_customer_id == "ctm_123"
    assert subscription.provider_subscription_id == "sub_123"

    assert subscription.provider_updated_at == EVENT_TIME

    assert subscription.status == SubscriptionStatus.ACTIVE.value
    assert subscription.billing_interval == BillingInterval.MONTHLY.value

    assert subscription.current_period_start == PERIOD_START
    assert subscription.current_period_end == PERIOD_END

    assert subscription.cancel_at_period_end is False

    assert subscription.past_due_since is None
    assert subscription.grace_period_ends_at is None

    assert event.organization_id == ORGANIZATION_ID
    assert event.subscription_id == subscription.id
    assert event.processed_at == EVENT_TIME


def test_creates_trial_subscription_with_trial_dates(
    db: Session,
) -> None:
    _seed_base(db)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    result = service.synchronize(
        state=_state(
            status=SubscriptionStatus.TRIALING.value,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    subscription = result.subscription

    assert subscription.status == SubscriptionStatus.TRIALING.value
    assert subscription.trial_started_at == PERIOD_START
    assert subscription.trial_ends_at == PERIOD_END


def test_creates_past_due_subscription_with_seven_day_grace(
    db: Session,
) -> None:
    _seed_base(db)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    result = service.synchronize(
        state=_state(
            status=SubscriptionStatus.PAST_DUE.value,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    subscription = result.subscription

    assert subscription.past_due_since == EVENT_TIME

    assert subscription.grace_period_ends_at == (
        EVENT_TIME
        + timedelta(
            days=PAST_DUE_GRACE_DAYS
        )
    )


def test_updates_existing_subscription(
    db: Session,
) -> None:
    _seed_base(db)

    db.add(
        _plan(
            plan_id=PROFESSIONAL_PLAN_ID,
            code=PlanCode.PROFESSIONAL.value,
        )
    )

    subscription = _existing_subscription(
        provider_updated_at=(
            EVENT_TIME
            - timedelta(hours=1)
        )
    )

    db.add(subscription)

    event = _billing_event(
        event_id="evt_update"
    )

    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    result = service.synchronize(
        state=_state(
            provider_customer_id="ctm_new",
            plan_code=PlanCode.PROFESSIONAL.value,
            billing_interval=BillingInterval.ANNUAL.value,
            cancel_at_period_end=True,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert result.applied is True
    assert result.stale is False

    assert subscription.plan_id == PROFESSIONAL_PLAN_ID
    assert subscription.provider_customer_id == "ctm_new"
    assert subscription.billing_interval == BillingInterval.ANNUAL.value
    assert subscription.cancel_at_period_end is True
    assert subscription.provider_updated_at == EVENT_TIME


def test_payment_recovery_clears_past_due_state(
    db: Session,
) -> None:
    _seed_base(db)

    subscription = _existing_subscription(
        status=SubscriptionStatus.PAST_DUE.value,
        provider_updated_at=(
            EVENT_TIME
            - timedelta(hours=1)
        ),
    )

    subscription.past_due_since = (
        EVENT_TIME
        - timedelta(days=1)
    )

    subscription.grace_period_ends_at = (
        EVENT_TIME
        + timedelta(days=6)
    )

    db.add(subscription)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    service.synchronize(
        state=_state(
            status=SubscriptionStatus.ACTIVE.value,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert subscription.status == SubscriptionStatus.ACTIVE.value
    assert subscription.past_due_since is None
    assert subscription.grace_period_ends_at is None


def test_repeated_past_due_event_does_not_extend_grace(
    db: Session,
) -> None:
    _seed_base(db)

    original_past_due = (
        EVENT_TIME
        - timedelta(days=1)
    )

    original_grace_end = (
        original_past_due
        + timedelta(days=PAST_DUE_GRACE_DAYS)
    )

    subscription = _existing_subscription(
        status=SubscriptionStatus.PAST_DUE.value,
        provider_updated_at=(
            EVENT_TIME
            - timedelta(hours=1)
        ),
    )

    subscription.past_due_since = original_past_due
    subscription.grace_period_ends_at = original_grace_end

    db.add(subscription)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    service.synchronize(
        state=_state(
            status=SubscriptionStatus.PAST_DUE.value,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert subscription.past_due_since == original_past_due
    assert subscription.grace_period_ends_at == original_grace_end


def test_stale_event_does_not_mutate_subscription(
    db: Session,
) -> None:
    _seed_base(db)

    subscription = _existing_subscription(
        provider_customer_id="ctm_current",
        provider_updated_at=EVENT_TIME,
    )

    db.add(subscription)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    result = service.synchronize(
        state=_state(
            provider_customer_id="ctm_old",
            status=SubscriptionStatus.CANCELED.value,
            current_period_start=None,
            current_period_end=None,
            event_occurred_at=(
                EVENT_TIME
                - timedelta(minutes=5)
            ),
        ),
        billing_event=event,
        processed_at=(
            EVENT_TIME
            + timedelta(minutes=1)
        ),
    )

    assert result.applied is False
    assert result.stale is True

    assert subscription.provider_customer_id == "ctm_current"
    assert subscription.status == SubscriptionStatus.ACTIVE.value
    assert subscription.provider_updated_at == EVENT_TIME

    assert event.organization_id == ORGANIZATION_ID
    assert event.subscription_id == subscription.id
    assert event.processed_at == (
        EVENT_TIME
        + timedelta(minutes=1)
    )


def test_equal_timestamp_is_treated_as_stale(
    db: Session,
) -> None:
    _seed_base(db)

    subscription = _existing_subscription(
        provider_updated_at=EVENT_TIME
    )

    db.add(subscription)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    result = service.synchronize(
        state=_state(
            status=SubscriptionStatus.CANCELED.value,
            current_period_start=None,
            current_period_end=None,
            event_occurred_at=EVENT_TIME,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert result.applied is False
    assert result.stale is True

    assert subscription.status == SubscriptionStatus.ACTIVE.value


def test_canceled_event_updates_subscription(
    db: Session,
) -> None:
    _seed_base(db)

    subscription = _existing_subscription(
        provider_updated_at=(
            EVENT_TIME
            - timedelta(hours=1)
        )
    )

    db.add(subscription)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    service.synchronize(
        state=_state(
            status=SubscriptionStatus.CANCELED.value,
            current_period_start=None,
            current_period_end=None,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert subscription.status == SubscriptionStatus.CANCELED.value
    assert subscription.current_period_start is None
    assert subscription.current_period_end is None


def test_read_only_event_updates_subscription(
    db: Session,
) -> None:
    _seed_base(db)

    subscription = _existing_subscription(
        provider_updated_at=(
            EVENT_TIME
            - timedelta(hours=1)
        )
    )

    db.add(subscription)

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    service.synchronize(
        state=_state(
            status=SubscriptionStatus.READ_ONLY.value,
            current_period_start=None,
            current_period_end=None,
        ),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert subscription.status == SubscriptionStatus.READ_ONLY.value


def test_unknown_organization_is_rejected(
    db: Session,
) -> None:
    db.add(
        _plan()
    )

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    with pytest.raises(
        PaddleOrganizationNotFoundError,
        match="unknown organization",
    ):
        service.synchronize(
            state=_state(),
            billing_event=event,
        )


def test_missing_plan_is_rejected(
    db: Session,
) -> None:
    db.add(
        _organization()
    )

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    with pytest.raises(
        PaddlePlanNotFoundError,
        match="plan does not exist",
    ):
        service.synchronize(
            state=_state(),
            billing_event=event,
        )


def test_inactive_plan_is_rejected(
    db: Session,
) -> None:
    db.add(
        _organization()
    )

    db.add(
        _plan(
            active=False
        )
    )

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    with pytest.raises(
        PaddlePlanNotFoundError,
        match="plan is inactive",
    ):
        service.synchronize(
            state=_state(),
            billing_event=event,
        )


def test_provider_subscription_cannot_move_between_organizations(
    db: Session,
) -> None:
    db.add(
        _organization()
    )

    db.add(
        _organization(
            organization_id=OTHER_ORGANIZATION_ID
        )
    )

    db.add(
        _plan()
    )

    db.add(
        _existing_subscription(
            organization_id=OTHER_ORGANIZATION_ID,
            provider_subscription_id="sub_123",
        )
    )

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    with pytest.raises(
        PaddleSubscriptionConflictError,
        match="another organization",
    ):
        service.synchronize(
            state=_state(),
            billing_event=event,
        )


def test_organization_cannot_switch_to_different_provider_subscription(
    db: Session,
) -> None:
    _seed_base(db)

    db.add(
        _existing_subscription(
            provider_subscription_id="sub_existing",
        )
    )

    event = _billing_event()
    db.add(event)
    db.flush()

    service = PaddleSubscriptionService(
        db
    )

    with pytest.raises(
        PaddleSubscriptionConflictError,
        match="different Paddle subscription",
    ):
        service.synchronize(
            state=_state(
                provider_subscription_id="sub_new",
            ),
            billing_event=event,
        )


def test_service_does_not_commit_transaction(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_base(db)

    event = _billing_event()
    db.add(event)
    db.flush()

    commit_calls = 0

    original_commit = db.commit

    def fake_commit() -> Any:
        nonlocal commit_calls
        commit_calls += 1

        return original_commit()

    monkeypatch.setattr(
        db,
        "commit",
        fake_commit,
    )

    service = PaddleSubscriptionService(
        db
    )

    service.synchronize(
        state=_state(),
        billing_event=event,
        processed_at=EVENT_TIME,
    )

    assert commit_calls == 0
