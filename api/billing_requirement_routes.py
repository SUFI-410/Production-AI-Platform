"""
Authenticated Invoice Preflight billing-requirements API routes.

The client supplies document IDs, while tenant ownership is always
derived from authenticated server-side organization state.
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
from sqlalchemy.orm import Session

from api.dependencies import (
    get_current_organization,
    get_db,
)
from api.schemas import (
    BillingRequirementsExtractRequest,
)
from rag.billing_requirements import (
    BillingRequirements,
    BillingRequirementsExtractionError,
    BillingRequirementsExtractor,
)
from rag.billing_requirements_service import (
    BillingRequirementsService,
    BillingRequirementsServiceError,
)
from rag.config import Config
from rag.document_storage import LocalDocumentStorage
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
    prefix="/billing-requirements",
    tags=["Billing Requirements"],
)


def get_billing_requirements_service(
) -> BillingRequirementsService:
    """
    Build the Invoice Preflight billing-requirements service.
    """

    storage = LocalDocumentStorage(
        Config.DOCUMENT_STORAGE_DIR
    )

    document_loader = TenantDocumentLoader(
        storage
    )

    extractor = BillingRequirementsExtractor()

    return BillingRequirementsService(
        document_loader=document_loader,
        extractor=extractor,
    )


@router.post(
    "/extract",
    response_model=BillingRequirements,
)
def extract_billing_requirements(
    request: BillingRequirementsExtractRequest,
    current_organization: Annotated[
        Organization,
        Depends(get_current_organization),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    service: Annotated[
        BillingRequirementsService,
        Depends(
            get_billing_requirements_service
        ),
    ],
) -> BillingRequirements:
    """
    Extract grounded billing requirements from tenant-owned documents.

    Document ownership is checked using the authenticated organization.
    Documents owned by another tenant are indistinguishable from missing
    documents to prevent cross-tenant information disclosure.
    """

    statement = select(
        DocumentRecord
    ).where(
        DocumentRecord.organization_id
        == current_organization.id,
        DocumentRecord.id.in_(
            request.document_ids
        ),
    )

    documents = list(
        db.scalars(
            statement
        ).all()
    )

    if len(documents) != len(
        request.document_ids
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

    ordered_documents = [
        documents_by_id[document_id]
        for document_id in request.document_ids
    ]

    try:
        return service.extract(
            ordered_documents
        )

    except BillingRequirementsServiceError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=str(exc),
        ) from exc

    except TenantDocumentLoadError:
        logger.exception(
            "Unable to load one or more tenant documents "
            "for billing requirements extraction."
        )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "One or more documents are not "
                "available for processing."
            ),
        ) from None

    except BillingRequirementsExtractionError:
        logger.exception(
            "Billing requirements extraction failed."
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Unable to extract billing requirements "
                "from the documents."
            ),
        ) from None
