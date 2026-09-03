"""
Authenticated tenant API routes for deterministic invoice preflight.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import (
    get_current_organization,
    get_db,
)
from api.schemas import InvoicePreflightRequest
from rag.billing_requirements import (
    BillingRequirementsExtractionError,
    BillingRequirementsExtractor,
)
from rag.billing_requirements_service import (
    BillingRequirementsService,
    BillingRequirementsServiceError,
)
from rag.billing_service import (
    BillingService,
    BillingServiceError,
    InvoiceCheckNotAllowedError,
    SubscriptionNotFoundError,
)
from rag.config import Config
from rag.document_storage import LocalDocumentStorage
from rag.invoice_facts import (
    InvoiceFactsExtractionError,
    InvoiceFactsExtractor,
)
from rag.invoice_preflight import InvoicePreflightResult
from rag.invoice_preflight_service import (
    InvoicePreflightService,
    InvoicePreflightServiceError,
)
from rag.models import (
    Document as DocumentRecord,
    Organization,
)
from rag.tenant_document_loader import (
    TenantDocumentLoadError,
    TenantDocumentLoader,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/invoice-preflight",
    tags=["Invoice Preflight"],
)


def get_invoice_preflight_service(
) -> InvoicePreflightService:
    """Build the Invoice Preflight application service."""

    storage = LocalDocumentStorage(
        Config.DOCUMENT_STORAGE_DIR
    )

    document_loader = TenantDocumentLoader(
        storage
    )

    billing_requirements_service = (
        BillingRequirementsService(
            document_loader=document_loader,
            extractor=BillingRequirementsExtractor(),
        )
    )

    return InvoicePreflightService(
        billing_requirements_service=(
            billing_requirements_service
        ),
        document_loader=document_loader,
        invoice_extractor=InvoiceFactsExtractor(),
    )


def get_billing_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> BillingService:
    """Build the request-scoped subscription billing service."""

    return BillingService(db)


@router.post(
    "/evaluate",
    response_model=InvoicePreflightResult,
)
def evaluate_invoice_preflight(
    request: InvoicePreflightRequest,
    current_organization: Annotated[
        Organization,
        Depends(get_current_organization),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    service: Annotated[
        InvoicePreflightService,
        Depends(get_invoice_preflight_service),
    ],
    billing_service: Annotated[
        BillingService,
        Depends(get_billing_service),
    ],
) -> InvoicePreflightResult:
    """
    Evaluate one invoice against tenant billing requirements.

    Every requested document is resolved using the authenticated
    organization. Documents belonging to another tenant are
    indistinguishable from missing documents.

    A successful preflight consumes one invoice-check allowance.
    Failed processing rolls back the usage reservation.
    """

    requested_document_ids = [
        *request.billing_document_ids,
        request.invoice_document_id,
    ]

    statement = select(
        DocumentRecord
    ).where(
        DocumentRecord.organization_id
        == current_organization.id,
        DocumentRecord.id.in_(
            requested_document_ids
        ),
    )

    try:
        documents = list(
            db.scalars(
                statement
            ).all()
        )
    except SQLAlchemyError:
        logger.exception(
            "Unable to resolve tenant documents "
            "for invoice preflight."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unable to resolve documents "
                "for invoice preflight."
            ),
        ) from None

    if len(documents) != len(
        requested_document_ids
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "One or more documents were not found."
            ),
        )

    documents_by_id = {
        document.id: document
        for document in documents
    }

    billing_documents = [
        documents_by_id[document_id]
        for document_id
        in request.billing_document_ids
    ]

    invoice_document = documents_by_id[
        request.invoice_document_id
    ]

    try:
        billing_service.consume_invoice_check(
            current_organization.id
        )

        result = service.evaluate(
            billing_documents,
            invoice_document,
        )

        db.commit()

        return result

    except SubscriptionNotFoundError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "A subscription or active trial "
                "is required to run invoice preflight."
            ),
        ) from exc

    except InvoiceCheckNotAllowedError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except BillingServiceError:
        db.rollback()

        logger.exception(
            "Unable to enforce invoice-preflight "
            "billing entitlement."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unable to verify subscription access."
            ),
        ) from None

    except (
        InvoicePreflightServiceError,
        BillingRequirementsServiceError,
    ) as exc:
        db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=str(exc),
        ) from exc

    except TenantDocumentLoadError:
        db.rollback()

        logger.exception(
            "Unable to load one or more tenant documents "
            "for invoice preflight."
        )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "One or more documents are not "
                "available for processing."
            ),
        ) from None

    except (
        BillingRequirementsExtractionError,
        InvoiceFactsExtractionError,
    ):
        db.rollback()

        logger.exception(
            "Invoice preflight extraction failed."
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Unable to extract invoice preflight data "
                "from the documents."
            ),
        ) from None

    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Unable to persist invoice-preflight usage."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unable to persist invoice-preflight usage."
            ),
        ) from None
