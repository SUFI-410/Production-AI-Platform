"""
Application service for extracting billing requirements from tenant documents.

This service connects durable tenant document records to the generic
document loader and the grounded Billing Requirements Extractor.
"""

from __future__ import annotations

from rag.billing_requirements import (
    BillingRequirements,
    BillingRequirementsExtractor,
)
from rag.models import (
    Document as DocumentRecord,
    DocumentType,
)
from rag.tenant_document_loader import TenantDocumentLoader


class BillingRequirementsServiceError(RuntimeError):
    """Raised when billing requirements cannot be extracted safely."""


_ALLOWED_REQUIREMENT_DOCUMENT_TYPES = {
    DocumentType.CONTRACT.value,
    DocumentType.SOW.value,
    DocumentType.PURCHASE_ORDER.value,
    DocumentType.BILLING_INSTRUCTIONS.value,
}


class BillingRequirementsService:
    """
    Extract grounded billing requirements from durable tenant documents.

    Only documents that can define customer billing requirements are
    accepted. Invoice and supporting-evidence documents belong to later
    Invoice Preflight comparison stages.
    """

    def __init__(
        self,
        document_loader: TenantDocumentLoader,
        extractor: BillingRequirementsExtractor,
    ) -> None:
        self.document_loader = document_loader
        self.extractor = extractor

    @staticmethod
    def _validate_documents(
        documents: list[DocumentRecord],
    ) -> None:
        """Validate records before loading any private tenant content."""

        if not documents:
            raise BillingRequirementsServiceError(
                "No billing requirement documents were provided."
            )

        unsupported_types = sorted(
            {
                document.document_type
                for document in documents
                if document.document_type
                not in _ALLOWED_REQUIREMENT_DOCUMENT_TYPES
            }
        )

        if unsupported_types:
            joined_types = ", ".join(
                unsupported_types
            )

            raise BillingRequirementsServiceError(
                "Unsupported billing requirement "
                f"document type(s): {joined_types}"
            )

    def extract(
        self,
        documents: list[DocumentRecord],
    ) -> BillingRequirements:
        """
        Load tenant files and extract verified billing requirements.
        """

        self._validate_documents(
            documents
        )

        loaded_documents = (
            self.document_loader.load_many(
                documents
            )
        )

        return self.extractor.extract(
            loaded_documents
        )
