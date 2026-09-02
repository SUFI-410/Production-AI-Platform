"""
Paddle billing webhook API routes.

This endpoint verifies Paddle webhook signatures before parsing
or persisting webhook payloads. Subscription state synchronization
is intentionally handled separately.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import get_db
from rag.config import Config
from rag.logger import get_logger
from rag.models import BillingEvent
from rag.paddle_webhooks import (
    PaddleWebhookVerificationError,
    verify_paddle_webhook,
)


logger = get_logger(__name__)


router = APIRouter(
    prefix="/webhooks",
    tags=["Billing"],
)


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
    Verify and persist a Paddle webhook event idempotently.

    The raw request body is verified before JSON parsing.
    This endpoint deliberately does not mutate subscription state yet.
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
        existing_event = db.scalar(
            select(BillingEvent).where(
                BillingEvent.provider == "paddle",
                BillingEvent.provider_event_id == event_id,
            )
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
        return {
            "status": "duplicate",
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
        db.commit()
    except IntegrityError:
        # Another worker may have inserted the same Paddle event
        # after our initial idempotency check.
        db.rollback()

        try:
            existing_event = db.scalar(
                select(BillingEvent).where(
                    BillingEvent.provider == "paddle",
                    BillingEvent.provider_event_id == event_id,
                )
            )
        except SQLAlchemyError:
            logger.exception(
                "Unable to resolve concurrent Paddle webhook."
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to process Paddle webhook.",
            ) from None

        if existing_event is not None:
            return {
                "status": "duplicate",
            }

        logger.exception(
            "Paddle webhook persistence failed with "
            "an unexpected integrity error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process Paddle webhook.",
        ) from None

    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Unable to persist Paddle webhook."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process Paddle webhook.",
        ) from None

    return {
        "status": "received",
    }
