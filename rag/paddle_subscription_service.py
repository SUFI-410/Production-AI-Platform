"""
Persist normalized Paddle subscription state.

This service synchronizes Paddle subscription events into the local
subscription table while protecting against stale/out-of-order events.
Transaction commit ownership remains with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.models import (
    BillingEvent,
    Organization,
    Plan,
    Subscription,
    SubscriptionStatus,
)
from rag.paddle_subscription_sync import PaddleSubscriptionState


PAST_DUE_GRACE_DAYS = 7


class PaddleSubscriptionServiceError(RuntimeError):
    """Base error for Paddle subscription persistence."""


class PaddleOrganizationNotFoundError(
    PaddleSubscriptionServiceError
):
    """Raised when Paddle references an unknown organization."""


class PaddlePlanNotFoundError(
    PaddleSubscriptionServiceError
):
    """Raised when the normalized Paddle plan cannot be resolved."""


class PaddleSubscriptionConflictError(
    PaddleSubscriptionServiceError
):
    """Raised when Paddle subscription ownership conflicts."""


@dataclass(frozen=True)
class PaddleSubscriptionSyncResult:
    """Outcome of synchronizing one Paddle subscription event."""

    subscription: Subscription
    applied: bool
    stale: bool


def _utc_now() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


def _as_utc(
    value: datetime,
) -> datetime:
    """
    Normalize database timestamps to timezone-aware UTC.

    SQLite may return offset-naive values even for timezone-enabled
    DateTime columns, while PostgreSQL preserves timezone information.
    """

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


class PaddleSubscriptionService:
    """Synchronize normalized Paddle state into local subscriptions."""

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session

    def _get_organization(
        self,
        organization_id: UUID,
    ) -> Organization:
        """Resolve the organization referenced by Paddle."""

        organization = self.session.get(
            Organization,
            organization_id,
        )

        if organization is None:
            raise PaddleOrganizationNotFoundError(
                "Paddle subscription references "
                "an unknown organization."
            )

        return organization

    def _get_plan(
        self,
        plan_code: str,
    ) -> Plan:
        """Resolve an active local plan by plan code."""

        plan = self.session.scalar(
            select(Plan).where(
                Plan.code == plan_code
            )
        )

        if plan is None:
            raise PaddlePlanNotFoundError(
                "Paddle subscription plan does not exist."
            )

        if not plan.is_active:
            raise PaddlePlanNotFoundError(
                "Paddle subscription plan is inactive."
            )

        return plan

    def _get_provider_subscription(
        self,
        provider_subscription_id: str,
    ) -> Subscription | None:
        """
        Resolve an existing subscription by Paddle subscription ID.

        The row is locked on PostgreSQL so concurrent webhook workers
        cannot independently reassign the same external subscription.
        """

        return self.session.scalar(
            select(Subscription)
            .where(
                Subscription.provider == "paddle",
                Subscription.provider_subscription_id
                == provider_subscription_id,
            )
            .with_for_update()
        )

    def _get_organization_subscription(
        self,
        organization_id: UUID,
    ) -> Subscription | None:
        """Resolve and lock the organization's subscription."""

        return self.session.scalar(
            select(Subscription)
            .where(
                Subscription.organization_id
                == organization_id
            )
            .with_for_update()
        )

    @staticmethod
    def _is_stale(
        subscription: Subscription,
        *,
        event_occurred_at: datetime,
    ) -> bool:
        """
        Return whether an event is older than or equal to applied state.

        Equal timestamps are also ignored so replayed provider events
        cannot mutate state twice.
        """

        if subscription.provider_updated_at is None:
            return False

        return (
            _as_utc(event_occurred_at)
            <= _as_utc(
                subscription.provider_updated_at
            )
        )

    @staticmethod
    def _set_past_due_state(
        subscription: Subscription,
        *,
        state: PaddleSubscriptionState,
    ) -> None:
        """Apply past-due timestamps without extending existing grace."""

        if (
            subscription.status
            != SubscriptionStatus.PAST_DUE.value
            or subscription.past_due_since is None
            or subscription.grace_period_ends_at is None
        ):
            subscription.past_due_since = (
                state.event_occurred_at
            )

            subscription.grace_period_ends_at = (
                state.event_occurred_at
                + timedelta(
                    days=PAST_DUE_GRACE_DAYS
                )
            )

    @staticmethod
    def _clear_past_due_state(
        subscription: Subscription,
    ) -> None:
        """Clear payment-failure grace state after recovery/transition."""

        subscription.past_due_since = None
        subscription.grace_period_ends_at = None

    @staticmethod
    def _apply_state(
        subscription: Subscription,
        *,
        plan: Plan,
        state: PaddleSubscriptionState,
    ) -> None:
        """Apply one non-stale normalized Paddle state."""

        if state.status == SubscriptionStatus.PAST_DUE.value:
            PaddleSubscriptionService._set_past_due_state(
                subscription,
                state=state,
            )
        else:
            PaddleSubscriptionService._clear_past_due_state(
                subscription
            )

        subscription.plan_id = plan.id
        subscription.provider = "paddle"

        subscription.provider_customer_id = (
            state.provider_customer_id
        )

        subscription.provider_subscription_id = (
            state.provider_subscription_id
        )

        subscription.provider_updated_at = (
            state.event_occurred_at
        )

        subscription.status = state.status
        subscription.billing_interval = state.billing_interval

        subscription.current_period_start = (
            state.current_period_start
        )

        subscription.current_period_end = (
            state.current_period_end
        )

        subscription.cancel_at_period_end = (
            state.cancel_at_period_end
        )

        if state.status == SubscriptionStatus.TRIALING.value:
            subscription.trial_started_at = (
                state.current_period_start
            )

            subscription.trial_ends_at = (
                state.current_period_end
            )

    @staticmethod
    def _link_event(
        billing_event: BillingEvent,
        *,
        subscription: Subscription,
        organization_id: UUID,
        processed_at: datetime,
    ) -> None:
        """Link the persisted provider event to synchronized state."""

        billing_event.organization_id = organization_id
        billing_event.subscription_id = subscription.id
        billing_event.processed_at = processed_at

    def synchronize(
        self,
        *,
        state: PaddleSubscriptionState,
        billing_event: BillingEvent,
        processed_at: datetime | None = None,
    ) -> PaddleSubscriptionSyncResult:
        """
        Synchronize one normalized Paddle subscription event.

        No commit occurs here. The caller owns the transaction.
        """

        effective_processed_at = _as_utc(
            processed_at or _utc_now()
        )

        self._get_organization(
            state.organization_id
        )

        plan = self._get_plan(
            state.plan_code
        )

        provider_subscription = (
            self._get_provider_subscription(
                state.provider_subscription_id
            )
        )

        if (
            provider_subscription is not None
            and provider_subscription.organization_id
            != state.organization_id
        ):
            raise PaddleSubscriptionConflictError(
                "Paddle subscription is already assigned "
                "to another organization."
            )

        organization_subscription = (
            self._get_organization_subscription(
                state.organization_id
            )
        )

        if organization_subscription is not None:
            existing_provider_id = (
                organization_subscription.provider_subscription_id
            )

            if (
                existing_provider_id is not None
                and existing_provider_id
                != state.provider_subscription_id
            ):
                raise PaddleSubscriptionConflictError(
                    "Organization already has a different "
                    "Paddle subscription."
                )

        subscription = (
            organization_subscription
            or provider_subscription
        )

        if subscription is None:
            subscription = Subscription(
                organization_id=state.organization_id,
                plan_id=plan.id,
                scheduled_plan_id=None,
                provider="paddle",
                provider_customer_id=(
                    state.provider_customer_id
                ),
                provider_subscription_id=(
                    state.provider_subscription_id
                ),
                provider_updated_at=(
                    state.event_occurred_at
                ),
                status=state.status,
                billing_interval=(
                    state.billing_interval
                ),
                current_period_start=(
                    state.current_period_start
                ),
                current_period_end=(
                    state.current_period_end
                ),
                cancel_at_period_end=(
                    state.cancel_at_period_end
                ),
                trial_started_at=None,
                trial_ends_at=None,
                read_only_until=None,
                past_due_since=None,
                grace_period_ends_at=None,
            )

            if (
                state.status
                == SubscriptionStatus.TRIALING.value
            ):
                subscription.trial_started_at = (
                    state.current_period_start
                )

                subscription.trial_ends_at = (
                    state.current_period_end
                )

            if (
                state.status
                == SubscriptionStatus.PAST_DUE.value
            ):
                subscription.past_due_since = (
                    state.event_occurred_at
                )

                subscription.grace_period_ends_at = (
                    state.event_occurred_at
                    + timedelta(
                        days=PAST_DUE_GRACE_DAYS
                    )
                )

            self.session.add(
                subscription
            )

            self.session.flush()

            self._link_event(
                billing_event,
                subscription=subscription,
                organization_id=state.organization_id,
                processed_at=effective_processed_at,
            )

            self.session.flush()

            return PaddleSubscriptionSyncResult(
                subscription=subscription,
                applied=True,
                stale=False,
            )

        if self._is_stale(
            subscription,
            event_occurred_at=state.event_occurred_at,
        ):
            self._link_event(
                billing_event,
                subscription=subscription,
                organization_id=state.organization_id,
                processed_at=effective_processed_at,
            )

            self.session.flush()

            return PaddleSubscriptionSyncResult(
                subscription=subscription,
                applied=False,
                stale=True,
            )

        self._apply_state(
            subscription,
            plan=plan,
            state=state,
        )

        self._link_event(
            billing_event,
            subscription=subscription,
            organization_id=state.organization_id,
            processed_at=effective_processed_at,
        )

        self.session.flush()

        return PaddleSubscriptionSyncResult(
            subscription=subscription,
            applied=True,
            stale=False,
        )
