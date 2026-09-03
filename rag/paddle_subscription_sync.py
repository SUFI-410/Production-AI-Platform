"""
Translate Paddle subscription webhook payloads into local billing state.

This module performs no database writes. It validates and normalizes
Paddle subscription data so persistence can be handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from rag.config import Config
from rag.models import (
    BillingInterval,
    PlanCode,
    SubscriptionStatus,
)


class PaddleSubscriptionSyncError(RuntimeError):
    """Raised when Paddle subscription data cannot be synchronized."""


@dataclass(frozen=True)
class PaddlePlanMapping:
    """Local plan represented by one Paddle price."""

    plan_code: str
    billing_interval: str


@dataclass(frozen=True)
class PaddleSubscriptionState:
    """Normalized state extracted from a Paddle subscription event."""

    organization_id: UUID
    provider_customer_id: str
    provider_subscription_id: str

    plan_code: str
    billing_interval: str

    status: str

    current_period_start: datetime | None
    current_period_end: datetime | None

    cancel_at_period_end: bool

    event_occurred_at: datetime


SUPPORTED_SUBSCRIPTION_EVENTS = {
    "subscription.created",
    "subscription.updated",
    "subscription.activated",
    "subscription.trialing",
    "subscription.past_due",
    "subscription.paused",
    "subscription.resumed",
    "subscription.canceled",
}


def _parse_datetime(
    value: Any,
    *,
    field_name: str,
) -> datetime:
    """Parse a Paddle RFC 3339 timestamp into UTC."""

    if not isinstance(value, str) or not value.strip():
        raise PaddleSubscriptionSyncError(
            f"Paddle {field_name} is missing."
        )

    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = (
            normalized[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )
    except ValueError as exc:
        raise PaddleSubscriptionSyncError(
            f"Paddle {field_name} is invalid."
        ) from exc

    if parsed.tzinfo is None:
        raise PaddleSubscriptionSyncError(
            f"Paddle {field_name} must include a timezone."
        )

    return parsed.astimezone(
        timezone.utc
    )


def _price_mapping() -> dict[str, PaddlePlanMapping]:
    """Return configured Paddle price-to-plan mappings."""

    configured = {
        Config.PADDLE_STARTER_MONTHLY_PRICE_ID: PaddlePlanMapping(
            plan_code=PlanCode.STARTER.value,
            billing_interval=BillingInterval.MONTHLY.value,
        ),
        Config.PADDLE_STARTER_ANNUAL_PRICE_ID: PaddlePlanMapping(
            plan_code=PlanCode.STARTER.value,
            billing_interval=BillingInterval.ANNUAL.value,
        ),
        Config.PADDLE_PROFESSIONAL_MONTHLY_PRICE_ID: PaddlePlanMapping(
            plan_code=PlanCode.PROFESSIONAL.value,
            billing_interval=BillingInterval.MONTHLY.value,
        ),
        Config.PADDLE_PROFESSIONAL_ANNUAL_PRICE_ID: PaddlePlanMapping(
            plan_code=PlanCode.PROFESSIONAL.value,
            billing_interval=BillingInterval.ANNUAL.value,
        ),
        Config.PADDLE_BUSINESS_MONTHLY_PRICE_ID: PaddlePlanMapping(
            plan_code=PlanCode.BUSINESS.value,
            billing_interval=BillingInterval.MONTHLY.value,
        ),
        Config.PADDLE_BUSINESS_ANNUAL_PRICE_ID: PaddlePlanMapping(
            plan_code=PlanCode.BUSINESS.value,
            billing_interval=BillingInterval.ANNUAL.value,
        ),
    }

    # Empty environment values must never become valid price IDs.
    return {
        price_id: mapping
        for price_id, mapping in configured.items()
        if price_id
    }


def _organization_id(
    data: dict[str, Any],
) -> UUID:
    """Read the internal organization ID from Paddle custom_data."""

    custom_data = data.get(
        "custom_data"
    )

    if not isinstance(
        custom_data,
        dict,
    ):
        raise PaddleSubscriptionSyncError(
            "Paddle subscription custom_data is missing."
        )

    raw_organization_id = custom_data.get(
        "organization_id"
    )

    if not isinstance(
        raw_organization_id,
        str,
    ):
        raise PaddleSubscriptionSyncError(
            "Paddle organization_id is missing."
        )

    try:
        return UUID(
            raw_organization_id
        )
    except ValueError as exc:
        raise PaddleSubscriptionSyncError(
            "Paddle organization_id is invalid."
        ) from exc


def _provider_id(
    data: dict[str, Any],
    field_name: str,
) -> str:
    """Return a required Paddle identifier."""

    value = data.get(
        field_name
    )

    if not isinstance(value, str) or not value.strip():
        raise PaddleSubscriptionSyncError(
            f"Paddle {field_name} is missing."
        )

    return value.strip()


def _plan_mapping(
    data: dict[str, Any],
) -> PaddlePlanMapping:
    """Resolve exactly one configured recurring Paddle price."""

    items = data.get(
        "items"
    )

    if not isinstance(items, list):
        raise PaddleSubscriptionSyncError(
            "Paddle subscription items are missing."
        )

    mappings = _price_mapping()

    matches: list[PaddlePlanMapping] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("recurring") is False:
            continue

        price = item.get(
            "price"
        )

        if not isinstance(price, dict):
            continue

        price_id = price.get(
            "id"
        )

        if not isinstance(price_id, str):
            continue

        mapping = mappings.get(
            price_id
        )

        if mapping is not None:
            matches.append(
                mapping
            )

    if not matches:
        raise PaddleSubscriptionSyncError(
            "Paddle subscription has no configured plan price."
        )

    unique_matches = {
        (
            mapping.plan_code,
            mapping.billing_interval,
        )
        for mapping in matches
    }

    if len(unique_matches) != 1:
        raise PaddleSubscriptionSyncError(
            "Paddle subscription contains multiple plan prices."
        )

    return matches[0]


def _local_status(
    paddle_status: Any,
) -> str:
    """Map Paddle subscription status to local subscription status."""

    if paddle_status == "active":
        return SubscriptionStatus.ACTIVE.value

    if paddle_status == "trialing":
        return SubscriptionStatus.TRIALING.value

    if paddle_status == "past_due":
        return SubscriptionStatus.PAST_DUE.value

    if paddle_status == "canceled":
        return SubscriptionStatus.CANCELED.value

    if paddle_status == "paused":
        return SubscriptionStatus.READ_ONLY.value

    raise PaddleSubscriptionSyncError(
        "Unsupported Paddle subscription status."
    )


def _current_period(
    data: dict[str, Any],
) -> tuple[
    datetime | None,
    datetime | None,
]:
    """Extract the Paddle current billing period."""

    period = data.get(
        "current_billing_period"
    )

    # Paddle reports null for paused and canceled subscriptions.
    if period is None:
        return (
            None,
            None,
        )

    if not isinstance(period, dict):
        raise PaddleSubscriptionSyncError(
            "Paddle current_billing_period is invalid."
        )

    starts_at = _parse_datetime(
        period.get("starts_at"),
        field_name="current_billing_period.starts_at",
    )

    ends_at = _parse_datetime(
        period.get("ends_at"),
        field_name="current_billing_period.ends_at",
    )

    if ends_at <= starts_at:
        raise PaddleSubscriptionSyncError(
            "Paddle billing period is invalid."
        )

    return (
        starts_at,
        ends_at,
    )


def _cancel_at_period_end(
    data: dict[str, Any],
) -> bool:
    """Return whether Paddle has a scheduled cancellation."""

    scheduled_change = data.get(
        "scheduled_change"
    )

    if scheduled_change is None:
        return False

    if not isinstance(
        scheduled_change,
        dict,
    ):
        raise PaddleSubscriptionSyncError(
            "Paddle scheduled_change is invalid."
        )

    return (
        scheduled_change.get("action")
        == "cancel"
    )


def parse_paddle_subscription_event(
    payload: Any,
) -> PaddleSubscriptionState:
    """
    Validate and normalize a Paddle subscription webhook payload.

    No database mutation occurs here.
    """

    if not isinstance(
        payload,
        dict,
    ):
        raise PaddleSubscriptionSyncError(
            "Paddle webhook payload is invalid."
        )

    event_type = payload.get(
        "event_type"
    )

    if event_type not in SUPPORTED_SUBSCRIPTION_EVENTS:
        raise PaddleSubscriptionSyncError(
            "Unsupported Paddle subscription event."
        )

    occurred_at = _parse_datetime(
        payload.get("occurred_at"),
        field_name="occurred_at",
    )

    data = payload.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):
        raise PaddleSubscriptionSyncError(
            "Paddle subscription data is missing."
        )

    mapping = _plan_mapping(
        data
    )

    period_start, period_end = _current_period(
        data
    )

    status = _local_status(
        data.get("status")
    )

    if status in {
        SubscriptionStatus.ACTIVE.value,
        SubscriptionStatus.PAST_DUE.value,
    } and (
        period_start is None
        or period_end is None
    ):
        raise PaddleSubscriptionSyncError(
            "Paddle active subscription is missing its billing period."
        )

    return PaddleSubscriptionState(
        organization_id=_organization_id(
            data
        ),
        provider_customer_id=_provider_id(
            data,
            "customer_id",
        ),
        provider_subscription_id=_provider_id(
            data,
            "id",
        ),
        plan_code=mapping.plan_code,
        billing_interval=mapping.billing_interval,
        status=status,
        current_period_start=period_start,
        current_period_end=period_end,
        cancel_at_period_end=_cancel_at_period_end(
            data
        ),
        event_occurred_at=occurred_at,
    )
