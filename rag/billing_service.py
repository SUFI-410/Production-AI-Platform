"""
Subscription entitlement and usage enforcement.

Paddle is the external billing source of truth. This service operates on the
locally synchronized subscription state and enforces application entitlements.
"""

from __future__ import annotations

import calendar
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.models import (
    Plan,
    Subscription,
    SubscriptionStatus,
    SubscriptionUsage,
)


class BillingServiceError(RuntimeError):
    """Base error for billing entitlement operations."""


class SubscriptionNotFoundError(BillingServiceError):
    """Raised when an organization has no subscription record."""


class InvoiceCheckNotAllowedError(BillingServiceError):
    """Raised when an organization cannot consume another invoice check."""


@dataclass(frozen=True)
class BillingEntitlement:
    """Resolved subscription access and usage state."""

    organization_id: uuid.UUID
    subscription_id: uuid.UUID
    plan_code: str
    subscription_status: str
    access_mode: str

    invoice_checks_used: int
    invoice_checks_limit: int
    invoice_checks_grace: int

    can_run_invoice_check: bool
    in_grace_buffer: bool

    period_start: datetime
    period_end: datetime


def _utc_now() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """
    Normalize a database datetime to timezone-aware UTC.

    SQLite may return offset-naive values for DateTime(timezone=True),
    while PostgreSQL preserves timezone-aware timestamps.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _add_months(
    value: datetime,
    months: int,
) -> datetime:
    """
    Add calendar months while preserving the billing-day anchor when possible.

    Example:
    January 31 + 1 month -> February 28/29.
    """

    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1

    last_day = calendar.monthrange(year, month)[1]
    day = min(value.day, last_day)

    return value.replace(
        year=year,
        month=month,
        day=day,
    )


class BillingService:
    """Resolve subscription entitlements and consume metered usage."""

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session

    def _get_subscription(
        self,
        organization_id: uuid.UUID,
        *,
        lock: bool = False,
    ) -> Subscription:
        """Return the organization's subscription."""

        statement = select(Subscription).where(
            Subscription.organization_id == organization_id
        )

        if lock:
            statement = statement.with_for_update()

        subscription = self.session.scalar(statement)

        if subscription is None:
            raise SubscriptionNotFoundError(
                "Organization has no subscription."
            )

        return subscription

    def _get_plan(
        self,
        plan_id: uuid.UUID,
    ) -> Plan:
        """Return the subscription plan."""

        plan = self.session.get(
            Plan,
            plan_id,
        )

        if plan is None:
            raise BillingServiceError(
                "Subscription plan does not exist."
            )

        if not plan.is_active:
            raise BillingServiceError(
                "Subscription plan is inactive."
            )

        return plan

    @staticmethod
    def _access_mode(
        subscription: Subscription,
        *,
        now: datetime,
    ) -> str:
        """
        Resolve effective application access.

        Values:
        - full
        - read_only
        """

        effective_now = _as_utc(now)
        status = subscription.status

        if status == SubscriptionStatus.ACTIVE.value:
            return "full"

        if status == SubscriptionStatus.TRIALING.value:
            if (
                subscription.trial_ends_at is not None
                and effective_now
                < _as_utc(subscription.trial_ends_at)
            ):
                return "full"

            return "read_only"

        if status == SubscriptionStatus.PAST_DUE.value:
            if (
                subscription.grace_period_ends_at is not None
                and effective_now
                < _as_utc(subscription.grace_period_ends_at)
            ):
                return "full"

            return "read_only"

        if status in {
            SubscriptionStatus.CANCELED.value,
            SubscriptionStatus.READ_ONLY.value,
        }:
            return "read_only"

        return "read_only"

    @staticmethod
    def _usage_period(
        subscription: Subscription,
        *,
        now: datetime,
    ) -> tuple[datetime, datetime]:
        """
        Resolve the current entitlement usage period.

        Trial usage spans the trial itself.

        Paid monthly and annual subscriptions receive monthly usage resets
        anchored to current_period_start, even when billing is annual.
        """

        effective_now = _as_utc(now)

        if subscription.status == SubscriptionStatus.TRIALING.value:
            if (
                subscription.trial_started_at is None
                or subscription.trial_ends_at is None
            ):
                raise BillingServiceError(
                    "Trial subscription is missing trial dates."
                )

            return (
                _as_utc(subscription.trial_started_at),
                _as_utc(subscription.trial_ends_at),
            )

        if subscription.current_period_start is None:
            raise BillingServiceError(
                "Paid subscription is missing current_period_start."
            )

        anchor = _as_utc(
            subscription.current_period_start
        )

        if effective_now < anchor:
            raise BillingServiceError(
                "Current time is before the subscription period."
            )

        period_start = anchor
        period_end = _add_months(
            anchor,
            1,
        )

        month_offset = 0

        while effective_now >= period_end:
            month_offset += 1

            period_start = _add_months(
                anchor,
                month_offset,
            )

            period_end = _add_months(
                anchor,
                month_offset + 1,
            )

        if subscription.current_period_end is not None:
            current_period_end = _as_utc(
                subscription.current_period_end
            )

            if period_end > current_period_end:
                period_end = current_period_end

        if period_end <= period_start:
            raise BillingServiceError(
                "Subscription usage period is invalid."
            )

        return (
            period_start,
            period_end,
        )

    def _get_or_create_usage(
        self,
        *,
        subscription: Subscription,
        organization_id: uuid.UUID,
        period_start: datetime,
        period_end: datetime,
        lock: bool = False,
    ) -> SubscriptionUsage:
        """Return the usage row for the current entitlement period."""

        statement = select(SubscriptionUsage).where(
            SubscriptionUsage.subscription_id == subscription.id,
            SubscriptionUsage.period_start == period_start,
        )

        if lock:
            statement = statement.with_for_update()

        usage = self.session.scalar(statement)

        if usage is not None:
            return usage

        usage = SubscriptionUsage(
            subscription_id=subscription.id,
            organization_id=organization_id,
            period_start=period_start,
            period_end=period_end,
            invoice_checks_used=0,
        )

        self.session.add(usage)
        self.session.flush()

        return usage

    def entitlement(
        self,
        organization_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> BillingEntitlement:
        """Return current billing access and invoice-check usage."""

        effective_now = _as_utc(
            now or _utc_now()
        )

        subscription = self._get_subscription(
            organization_id
        )

        plan = self._get_plan(
            subscription.plan_id
        )

        period_start, period_end = self._usage_period(
            subscription,
            now=effective_now,
        )

        usage = self._get_or_create_usage(
            subscription=subscription,
            organization_id=organization_id,
            period_start=period_start,
            period_end=period_end,
        )

        access_mode = self._access_mode(
            subscription,
            now=effective_now,
        )

        normal_limit = plan.invoice_checks_limit
        hard_limit = (
            plan.invoice_checks_limit
            + plan.invoice_checks_grace
        )

        can_run = (
            access_mode == "full"
            and usage.invoice_checks_used < hard_limit
        )

        return BillingEntitlement(
            organization_id=organization_id,
            subscription_id=subscription.id,
            plan_code=plan.code,
            subscription_status=subscription.status,
            access_mode=access_mode,
            invoice_checks_used=usage.invoice_checks_used,
            invoice_checks_limit=normal_limit,
            invoice_checks_grace=plan.invoice_checks_grace,
            can_run_invoice_check=can_run,
            in_grace_buffer=(
                usage.invoice_checks_used >= normal_limit
                and usage.invoice_checks_used < hard_limit
            ),
            period_start=period_start,
            period_end=period_end,
        )

    def consume_invoice_check(
        self,
        organization_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> BillingEntitlement:
        """
        Atomically consume one invoice check.

        The usage row is locked before incrementing so concurrent requests
        cannot intentionally exceed the plan's hard limit on PostgreSQL.
        """

        effective_now = _as_utc(
            now or _utc_now()
        )

        subscription = self._get_subscription(
            organization_id,
            lock=True,
        )

        plan = self._get_plan(
            subscription.plan_id
        )

        access_mode = self._access_mode(
            subscription,
            now=effective_now,
        )

        if access_mode != "full":
            raise InvoiceCheckNotAllowedError(
                "Subscription is read-only."
            )

        period_start, period_end = self._usage_period(
            subscription,
            now=effective_now,
        )

        usage = self._get_or_create_usage(
            subscription=subscription,
            organization_id=organization_id,
            period_start=period_start,
            period_end=period_end,
            lock=True,
        )

        hard_limit = (
            plan.invoice_checks_limit
            + plan.invoice_checks_grace
        )

        if usage.invoice_checks_used >= hard_limit:
            raise InvoiceCheckNotAllowedError(
                "Invoice-check allowance has been exhausted."
            )

        usage.invoice_checks_used += 1

        self.session.flush()

        return BillingEntitlement(
            organization_id=organization_id,
            subscription_id=subscription.id,
            plan_code=plan.code,
            subscription_status=subscription.status,
            access_mode=access_mode,
            invoice_checks_used=usage.invoice_checks_used,
            invoice_checks_limit=plan.invoice_checks_limit,
            invoice_checks_grace=plan.invoice_checks_grace,
            can_run_invoice_check=(
                usage.invoice_checks_used < hard_limit
            ),
            in_grace_buffer=(
                usage.invoice_checks_used
                >= plan.invoice_checks_limit
                and usage.invoice_checks_used < hard_limit
            ),
            period_start=period_start,
            period_end=period_end,
        )
