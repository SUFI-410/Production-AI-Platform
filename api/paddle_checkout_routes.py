"""
Authenticated Paddle checkout API routes.
"""

from __future__ import annotations

from typing import Annotated

from api.dependencies import get_current_organization, get_db
from api.schemas import (
    BillingStatusResponse,
    PaddleCheckoutRequest,
    PaddleCheckoutResponse,
    PaddlePortalResponse,
)
from fastapi import APIRouter, Depends, HTTPException, Response, status
from rag.billing_service import (
    BillingService,
    BillingServiceError,
    SubscriptionNotFoundError,
)
from rag.models import Organization
from rag.paddle_checkout import (
    PaddleCheckoutAPIError,
    PaddleCheckoutConfigurationError,
    PaddleCheckoutService,
)
from rag.paddle_portal import (
    PaddlePortalAPIError,
    PaddlePortalConfigurationError,
    PaddlePortalError,
    PaddlePortalService,
    PaddlePortalUnavailableError,
)
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/billing",
    tags=["Billing"],
)


def get_billing_service(
    db: Annotated[Session, Depends(get_db)],
) -> BillingService:
    """Return the request-scoped local billing service."""

    return BillingService(db)


@router.get(
    "/status",
    response_model=BillingStatusResponse,
    status_code=status.HTTP_200_OK,
)
def get_billing_status(
    organization: Annotated[
        Organization,
        Depends(get_current_organization),
    ],
    service: Annotated[
        BillingService,
        Depends(get_billing_service),
    ],
) -> BillingStatusResponse:
    """Return the authenticated organization's billing status."""

    try:
        result = service.status(organization.id)
    except SubscriptionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization has no subscription.",
        ) from None
    except BillingServiceError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to resolve billing status.",
        ) from None

    return BillingStatusResponse(**result.__dict__)


@router.post(
    "/checkout",
    response_model=PaddleCheckoutResponse,
    status_code=status.HTTP_200_OK,
)
def create_paddle_checkout(
    request: PaddleCheckoutRequest,
    organization: Annotated[
        Organization,
        Depends(get_current_organization),
    ],
) -> PaddleCheckoutResponse:
    """
    Create a Paddle checkout transaction for the authenticated organization.

    The organization ID is always derived from authentication and is never
    accepted from the request body.
    """

    service = PaddleCheckoutService()

    try:
        result = service.create_checkout(
            organization_id=organization.id,
            plan_code=request.plan_code,
            billing_interval=request.billing_interval,
        )
    except PaddleCheckoutConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Billing checkout is not configured.",
        ) from None

    except PaddleCheckoutAPIError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to create billing checkout.",
        ) from None

    return PaddleCheckoutResponse(
        transaction_id=result.transaction_id,
        checkout_url=result.checkout_url,
    )


def get_portal_service(
    db: Annotated[Session, Depends(get_db)],
) -> PaddlePortalService:
    """Return a request-scoped portal service."""
    return PaddlePortalService(db)


@router.post("/portal", response_model=PaddlePortalResponse)
def create_paddle_portal(
    response: Response,
    organization: Annotated[Organization, Depends(get_current_organization)],
    service: Annotated[PaddlePortalService, Depends(get_portal_service)],
) -> PaddlePortalResponse:
    """Create a fresh portal link for the authenticated organization."""
    try:
        url = service.create_session(organization.id)
    except PaddlePortalUnavailableError:
        raise HTTPException(404, "No linked Paddle subscription.") from None
    except PaddlePortalConfigurationError:
        raise HTTPException(503, "Billing portal is not configured.") from None
    except PaddlePortalAPIError:
        raise HTTPException(502, "Unable to open billing portal.") from None
    except PaddlePortalError:
        raise HTTPException(500, "Unable to resolve billing account.") from None
    response.headers["Cache-Control"] = "no-store"
    return PaddlePortalResponse(portal_url=url)
