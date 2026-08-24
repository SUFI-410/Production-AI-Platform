"""
Application service for running deterministic tenant invoice preflight.

This service orchestrates billing-requirements extraction, invoice-facts
extraction, and deterministic comparison without allowing cross-tenant
document mixing.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

from rag.billing_requirements_service import (
    BillingRequirementsService,
)
from rag.invoice_facts import InvoiceFactsExtractor
from rag.invoice_preflight import (
    InvoicePreflightEngine,
    InvoicePreflightResult,
)
from rag.models import (
    Document as DocumentRecord,
    DocumentType,
)
from rag.tenant_document_loader import TenantDocumentLoader


logger = logging.getLogger(__name__)


class InvoicePreflightServiceError(RuntimeError):
    """Raised when an invoice preflight cannot be run safely."""


class InvoicePreflightService:
    """
    Run invoice preflight using durable tenant-owned documents.

    Billing documents define the expected billing requirements. The
    invoice document supplies factual invoice values. Final readiness
    decisions are made only by the deterministic Python comparison
    engine.
    """

    def __init__(
        self,
        billing_requirements_service: BillingRequirementsService,
        document_loader: TenantDocumentLoader,
        invoice_extractor: InvoiceFactsExtractor,
    ) -> None:
        self.billing_requirements_service = (
            billing_requirements_service
        )
        self.document_loader = document_loader
        self.invoice_extractor = invoice_extractor

    @staticmethod
    def _validate_documents(
        billing_documents: list[DocumentRecord],
        invoice_document: DocumentRecord,
    ) -> None:
        """
        Validate all document records before loading tenant content.

        Cross-tenant validation happens here instead of relying only on
        separate loader calls. This prevents billing documents from one
        organization being compared with an invoice from another.
        """

        if not billing_documents:
            raise InvoicePreflightServiceError(
                "No billing requirement documents were provided."
            )

        if (
            invoice_document.document_type
            != DocumentType.INVOICE.value
        ):
            raise InvoicePreflightServiceError(
                "The invoice document must have document type invoice."
            )

        organization_ids = {
            invoice_document.organization_id,
            *(
                document.organization_id
                for document in billing_documents
            ),
        }

        if len(organization_ids) != 1:
            raise InvoicePreflightServiceError(
                "Billing requirement and invoice documents "
                "must belong to the same organization."
            )

    def evaluate(
        self,
        billing_documents: list[DocumentRecord],
        invoice_document: DocumentRecord,
    ) -> InvoicePreflightResult:
        """
        Extract facts and return deterministic payment readiness.

        No language model decides PASS, WARNING, BLOCKER, or the overall
        payment-readiness state.
        """

        self._validate_documents(
            billing_documents,
            invoice_document,
        )

        loaded_invoice_documents = (
            self.document_loader.load(
                invoice_document
            )
        )

        started_at = monotonic()

        # These extractions are independent. Running them concurrently
        # bounds normal endpoint latency to the slower model call rather
        # than the sum of both model calls.
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="invoice-preflight",
        ) as executor:
            billing_future = executor.submit(
                self.billing_requirements_service.extract,
                billing_documents,
            )
            invoice_future = executor.submit(
                self.invoice_extractor.extract,
                loaded_invoice_documents,
            )

            billing_requirements = billing_future.result()
            invoice_facts = invoice_future.result()

        logger.info(
            "Invoice preflight extraction completed in %.2f seconds.",
            monotonic() - started_at,
        )

        return InvoicePreflightEngine.evaluate(
            billing_requirements,
            invoice_facts,
        )
