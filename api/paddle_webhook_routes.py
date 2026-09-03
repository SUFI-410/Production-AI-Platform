"""
Paddle billing webhook API routes.

Webhook signatures are verified before JSON parsing. Events are persisted
idempotently and supported subscription events are synchronized into the
local subscription state in the same database transaction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import get_db
from rag.config import Config
from rag.logger import get_logger
from rag.models import BillingEvent
from rag.paddle_subscription_service import (
    PaddleSubscriptionService,
    PaddleSubscriptionServiceError,
)
from rag.paddle_subscription_sync import (
    SUPPORTED_SUBSCRIPTION_EVENTS,
    PaddleSubscriptionSyncError,
    parse_paddle_subscription_event,
)
from rag.paddle_webhooks import (
    PaddleWebhookVerificationError,
    verify_paddle_webhook,
)


logger = get_logger(__name__)


router = APIRouter(
    prefix="/webhooks",
    tags=["Billing"],
)


def _utc_now() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


def _validate_event_payload(
    payload: Any,
) -> tuple[str, str]:
    """
    Validate the minimum Paddle webhook envelope.

    Returns:
        Tuple containing provider event ID and event type.
    """

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Paddle webhook payload.",
        )

    event_id = payload.get("event_id")
    event_type = payload.get("event_type")

    if (
        not isinstance(event_id, str)
        or not event_id.strip()
        or not isinstance(event_type, str)
        or not event_type.strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Paddle webhook payload.",
        )

    return (
        event_id.strip(),
        event_type.strip(),
    )


def _get_existing_event(
    db: Session,
    *,
    event_id: str,
) -> BillingEvent | None:
    """Return an already-persisted Paddle event."""

    return db.scalar(
        select(BillingEvent).where(
            BillingEvent.provider == "paddle",
            BillingEvent.provider_event_id == event_id,
        )
    )


def _process_billing_event(
    db: Session,
    *,
    billing_event: BillingEvent,
) -> None:
    """
    Process one persisted Paddle billing event.

    Unsupported Paddle event types are intentionally treated as processed
    after persistence because they require no local subscription mutation.

    Supported subscription events are normalized and synchronized by the
    Paddle subscription service.

    The caller owns the transaction commit.
    """

    if (
        billing_event.event_type
        not in SUPPORTED_SUBSCRIPTION_EVENTS
    ):
        billing_event.processed_at = _utc_now()
        return

    state = parse_paddle_subscription_event(
        billing_event.payload
    )

    service = PaddleSubscriptionService(
        db
    )

    service.synchronize(
        state=state,
        billing_event=billing_event,
    )


def _process_and_commit(
    db: Session,
    *,
    billing_event: BillingEvent,
) -> None:
    """Process an event and atomically commit its resulting state."""

    _process_billing_event(
        db,
        billing_event=billing_event,
    )

    db.commit()


@router.post(
    "/paddle",
    status_code=status.HTTP_200_OK,
)
async def receive_paddle_webhook(
    request: Request,
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> dict[str, str]:
    """
    Verify, persist, and process a Paddle webhook idempotently.

    A processed duplicate returns success immediately.

    A previously persisted but unprocessed event is retried instead of
    being discarded as a duplicate.
    """

    raw_body = await request.body()

    signature_header = request.headers.get(
        "Paddle-Signature",
        "",
    )

    try:
        verify_paddle_webhook(
            raw_body=raw_body,
            signature_header=signature_header,
            secret=Config.PADDLE_WEBHOOK_SECRET,
            tolerance_seconds=(
                Config.PADDLE_WEBHOOK_TOLERANCE_SECONDS
            ),
        )
    except PaddleWebhookVerificationError:
        logger.warning(
            "Rejected Paddle webhook with invalid signature."
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Paddle webhook signature.",
        ) from None

    try:
        payload = json.loads(
            raw_body.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Paddle webhook payload.",
        ) from None

    event_id, event_type = _validate_event_payload(
        payload
    )

    try:
        existing_event = _get_existing_event(
            db,
            event_id=event_id,
        )
    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Unable to check Paddle webhook idempotency."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process Paddle webhook.",
        ) from None

    if existing_event is not None:
        if existing_event.processed_at is not None:
            return {
                "status": "duplicate",
            }

        try:
            _process_and_commit(
                db,
                billing_event=existing_event,
            )
        except (
            PaddleSubscriptionSyncError,
            PaddleSubscriptionServiceError,
        ):
            db.rollback()

            logger.exception(
                "Unable to synchronize previously persisted "
                "Paddle subscription event."
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to process Paddle webhook.",
            ) from None

        except SQLAlchemyError:
            db.rollback()

            logger.exception(
                "Database failure while retrying "
                "Paddle webhook processing."
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to process Paddle webhook.",
            ) from None

        return {
            "status": "received",
        }

    billing_event = BillingEvent(
        provider="paddle",
        provider_event_id=event_id,
        event_type=event_type,
        payload=payload,
    )

    db.add(
        billing_event
    )

    try:
        _process_and_commit(
            db,
            billing_event=billing_event,
        )

    except IntegrityError:
        # A second worker may have inserted this event between our
        # idempotency query and transaction completion.
        db.rollback()

        try:
            existing_event = _get_existing_event(
                db,
                event_id=event_id,
            )
        except SQLAlchemyError:
            db.rollback()

            logger.exception(
                "Unable to resolve concurrent Paddle webhook."
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to process Paddle webhook.",
            ) from None

        if existing_event is None:
            logger.exception(
                "Paddle webhook transaction failed with "
                "an unrelated integrity error."
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to process Paddle webhook.",
            ) from None

        if existing_event.processed_at is not None:
            return {
                "status": "duplicate",
            }

        try:
            _process_and_commit(
                db,
                billing_event=existing_event,
            )
        except (
            PaddleSubscriptionSyncError,
            PaddleSubscriptionServiceError,
            SQLAlchemyError,
        ):
            db.rollback()

            logger.exception(
                "Unable to process concurrent persisted "
                "Paddle webhook."
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to process Paddle webhook.",
            ) from None

        return {
            "status": "received",
        }

    except (
        PaddleSubscriptionSyncError,
        PaddleSubscriptionServiceError,
    ):
        db.rollback()

        logger.exception(
            "Unable to synchronize Paddle subscription event."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process Paddle webhook.",
        ) from None

    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Database failure while processing Paddle webhook."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process Paddle webhook.",
        ) from None

    return {
        "status": "received",
    }
