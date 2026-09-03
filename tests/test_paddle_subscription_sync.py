"""Tests for Paddle subscription webhook normalization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import pytest

from rag.config import Config
from rag.models import (
    BillingInterval,
    PlanCode,
    SubscriptionStatus,
)
from rag.paddle_subscription_sync import (
    PaddleSubscriptionSyncError,
    parse_paddle_subscription_event,
)


ORGANIZATION_ID = UUID(
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
)

OCCURRED_AT = "2026-09-02T06:30:00Z"
PERIOD_START = "2026-09-01T00:00:00Z"
PERIOD_END = "2026-10-01T00:00:00Z"

_UNSET = object()


@pytest.fixture(autouse=True)
def configure_price_ids():
    original_values = {
        "starter_monthly": Config.PADDLE_STARTER_MONTHLY_PRICE_ID,
        "starter_annual": Config.PADDLE_STARTER_ANNUAL_PRICE_ID,
        "professional_monthly": (
            Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID
        ),
        "professional_annual": (
            Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID
        ),
        "business_monthly": Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID,
        "business_annual": Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID,
    }

    Config.PADDLE_STARTER_MONTHLY_PRICE_ID = "pri_starter_monthly"
    Config.PADDLE_STARTER_ANNUAL_PRICE_ID = "pri_starter_annual"

    Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID = (
        "pri_professional_monthly"
    )
    Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID = (
        "pri_professional_annual"
    )

    Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID = "pri_business_monthly"
    Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID = "pri_business_annual"

    yield

    Config.PADDLE_STARTER_MONTHLY_PRICE_ID = original_values[
        "starter_monthly"
    ]
    Config.PADDLE_STARTER_ANNUAL_PRICE_ID = original_values[
        "starter_annual"
    ]

    Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID = original_values[
        "professional_monthly"
    ]
    Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID = original_values[
        "professional_annual"
    ]

    Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID = original_values[
        "business_monthly"
    ]
    Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID = original_values[
        "business_annual"
    ]


def _payload(
    *,
    event_type: str = "subscription.updated",
    paddle_status: str = "active",
    price_id: str = "pri_starter_monthly",
    custom_data: Any = _UNSET,
    items: Any = _UNSET,
    current_billing_period: Any = _UNSET,
    scheduled_change: Any = None,
    occurred_at: Any = OCCURRED_AT,
) -> dict[str, Any]:
    if custom_data is _UNSET:
        custom_data = {
            "organization_id": str(
                ORGANIZATION_ID
            )
        }

    if items is _UNSET:
        items = [
            {
                "recurring": True,
                "price": {
                    "id": price_id,
                },
            }
        ]

    if current_billing_period is _UNSET:
        current_billing_period = {
            "starts_at": PERIOD_START,
            "ends_at": PERIOD_END,
        }

    return {
        "event_id": "evt_123",
        "event_type": event_type,
        "occurred_at": occurred_at,
        "data": {
            "id": "sub_123",
            "customer_id": "ctm_123",
            "status": paddle_status,
            "custom_data": custom_data,
            "items": items,
            "current_billing_period": current_billing_period,
            "scheduled_change": scheduled_change,
        },
    }


@pytest.mark.parametrize(
    (
        "price_id",
        "expected_plan",
        "expected_interval",
    ),
    [
        (
            "pri_starter_monthly",
            PlanCode.STARTER.value,
            BillingInterval.MONTHLY.value,
        ),
        (
            "pri_starter_annual",
            PlanCode.STARTER.value,
            BillingInterval.ANNUAL.value,
        ),
        (
            "pri_professional_monthly",
            PlanCode.PROFESSIONAL.value,
            BillingInterval.MONTHLY.value,
        ),
        (
            "pri_professional_annual",
            PlanCode.PROFESSIONAL.value,
            BillingInterval.ANNUAL.value,
        ),
        (
            "pri_business_monthly",
            PlanCode.BUSINESS.value,
            BillingInterval.MONTHLY.value,
        ),
        (
            "pri_business_annual",
            PlanCode.BUSINESS.value,
            BillingInterval.ANNUAL.value,
        ),
    ],
)
def test_maps_configured_price_to_local_plan(
    price_id: str,
    expected_plan: str,
    expected_interval: str,
) -> None:
    state = parse_paddle_subscription_event(
        _payload(
            price_id=price_id
        )
    )

    assert state.organization_id == ORGANIZATION_ID
    assert state.provider_customer_id == "ctm_123"
    assert state.provider_subscription_id == "sub_123"

    assert state.plan_code == expected_plan
    assert state.billing_interval == expected_interval

    assert state.status == SubscriptionStatus.ACTIVE.value

    assert state.current_period_start == datetime(
        2026,
        9,
        1,
        tzinfo=timezone.utc,
    )

    assert state.current_period_end == datetime(
        2026,
        10,
        1,
        tzinfo=timezone.utc,
    )

    assert state.event_occurred_at == datetime(
        2026,
        9,
        2,
        6,
        30,
        tzinfo=timezone.utc,
    )

    assert state.cancel_at_period_end is False


@pytest.mark.parametrize(
    (
        "paddle_status",
        "expected_status",
    ),
    [
        (
            "active",
            SubscriptionStatus.ACTIVE.value,
        ),
        (
            "trialing",
            SubscriptionStatus.TRIALING.value,
        ),
        (
            "past_due",
            SubscriptionStatus.PAST_DUE.value,
        ),
        (
            "canceled",
            SubscriptionStatus.CANCELED.value,
        ),
        (
            "paused",
            SubscriptionStatus.READ_ONLY.value,
        ),
    ],
)
def test_maps_paddle_status(
    paddle_status: str,
    expected_status: str,
) -> None:
    period: Any = {
        "starts_at": PERIOD_START,
        "ends_at": PERIOD_END,
    }

    if paddle_status in {
        "canceled",
        "paused",
    }:
        period = None

    state = parse_paddle_subscription_event(
        _payload(
            paddle_status=paddle_status,
            current_billing_period=period,
        )
    )

    assert state.status == expected_status


def test_scheduled_cancellation_sets_cancel_at_period_end(
) -> None:
    state = parse_paddle_subscription_event(
        _payload(
            scheduled_change={
                "action": "cancel",
                "effective_at": PERIOD_END,
            }
        )
    )

    assert state.status == SubscriptionStatus.ACTIVE.value
    assert state.cancel_at_period_end is True


def test_non_cancel_scheduled_change_does_not_set_cancellation(
) -> None:
    state = parse_paddle_subscription_event(
        _payload(
            scheduled_change={
                "action": "pause",
                "effective_at": PERIOD_END,
            }
        )
    )

    assert state.cancel_at_period_end is False


def test_unknown_price_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="no configured plan price",
    ):
        parse_paddle_subscription_event(
            _payload(
                price_id="pri_unknown"
            )
        )


def test_multiple_different_plan_prices_are_rejected(
) -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="multiple plan prices",
    ):
        parse_paddle_subscription_event(
            _payload(
                items=[
                    {
                        "recurring": True,
                        "price": {
                            "id": "pri_starter_monthly",
                        },
                    },
                    {
                        "recurring": True,
                        "price": {
                            "id": "pri_business_monthly",
                        },
                    },
                ]
            )
        )


def test_duplicate_same_plan_price_is_allowed() -> None:
    state = parse_paddle_subscription_event(
        _payload(
            items=[
                {
                    "recurring": True,
                    "price": {
                        "id": "pri_starter_monthly",
                    },
                },
                {
                    "recurring": True,
                    "price": {
                        "id": "pri_starter_monthly",
                    },
                },
            ]
        )
    )

    assert state.plan_code == PlanCode.STARTER.value
    assert state.billing_interval == BillingInterval.MONTHLY.value


def test_non_recurring_items_do_not_determine_plan() -> None:
    state = parse_paddle_subscription_event(
        _payload(
            items=[
                {
                    "recurring": False,
                    "price": {
                        "id": "pri_business_annual",
                    },
                },
                {
                    "recurring": True,
                    "price": {
                        "id": "pri_starter_monthly",
                    },
                },
            ]
        )
    )

    assert state.plan_code == PlanCode.STARTER.value


def test_invalid_organization_id_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="organization_id is invalid",
    ):
        parse_paddle_subscription_event(
            _payload(
                custom_data={
                    "organization_id": "not-a-uuid"
                }
            )
        )


def test_missing_custom_data_is_rejected() -> None:
    payload = _payload(
        custom_data=None
    )

    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="custom_data is missing",
    ):
        parse_paddle_subscription_event(
            payload
        )


def test_missing_customer_id_is_rejected() -> None:
    payload = _payload()
    payload["data"]["customer_id"] = ""

    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="customer_id is missing",
    ):
        parse_paddle_subscription_event(
            payload
        )


def test_missing_subscription_id_is_rejected() -> None:
    payload = _payload()
    payload["data"]["id"] = ""

    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="id is missing",
    ):
        parse_paddle_subscription_event(
            payload
        )


def test_unsupported_event_type_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="Unsupported Paddle subscription event",
    ):
        parse_paddle_subscription_event(
            _payload(
                event_type="transaction.completed"
            )
        )


def test_unsupported_status_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="Unsupported Paddle subscription status",
    ):
        parse_paddle_subscription_event(
            _payload(
                paddle_status="unknown"
            )
        )


def test_invalid_occurred_at_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="occurred_at is invalid",
    ):
        parse_paddle_subscription_event(
            _payload(
                occurred_at="not-a-date"
            )
        )


def test_occurred_at_without_timezone_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="must include a timezone",
    ):
        parse_paddle_subscription_event(
            _payload(
                occurred_at="2026-09-02T06:30:00"
            )
        )


def test_invalid_period_start_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match=(
            "current_billing_period.starts_at "
            "is invalid"
        ),
    ):
        parse_paddle_subscription_event(
            _payload(
                current_billing_period={
                    "starts_at": "invalid",
                    "ends_at": PERIOD_END,
                }
            )
        )


def test_period_end_must_be_after_start() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="billing period is invalid",
    ):
        parse_paddle_subscription_event(
            _payload(
                current_billing_period={
                    "starts_at": PERIOD_END,
                    "ends_at": PERIOD_START,
                }
            )
        )


@pytest.mark.parametrize(
    "paddle_status",
    [
        "active",
        "past_due",
    ],
)
def test_processing_status_requires_current_billing_period(
    paddle_status: str,
) -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="missing its billing period",
    ):
        parse_paddle_subscription_event(
            _payload(
                paddle_status=paddle_status,
                current_billing_period=None,
            )
        )


def test_canceled_subscription_can_have_no_current_period(
) -> None:
    state = parse_paddle_subscription_event(
        _payload(
            paddle_status="canceled",
            current_billing_period=None,
        )
    )

    assert state.status == SubscriptionStatus.CANCELED.value
    assert state.current_period_start is None
    assert state.current_period_end is None


def test_paused_subscription_can_have_no_current_period(
) -> None:
    state = parse_paddle_subscription_event(
        _payload(
            paddle_status="paused",
            current_billing_period=None,
        )
    )

    assert state.status == SubscriptionStatus.READ_ONLY.value
    assert state.current_period_start is None
    assert state.current_period_end is None


def test_invalid_scheduled_change_is_rejected() -> None:
    with pytest.raises(
        PaddleSubscriptionSyncError,
        match="scheduled_change is invalid",
    ):
        parse_paddle_subscription_event(
            _payload(
                scheduled_change="cancel"
            )
        )
