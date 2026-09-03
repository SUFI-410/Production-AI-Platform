"""
Authenticated Paddle checkout API routes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_organization
from api.schemas import (
    PaddleCheckoutRequest,
    PaddleCheckoutResponse,
)
from rag.models import Organization
from rag.paddle_checkout import (
    PaddleCheckoutAPIError,
    PaddleCheckoutConfigurationError,
    PaddleCheckoutService,
)


router = APIRouter(
    prefix="/billing",
    tags=["Billing"],
)


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
