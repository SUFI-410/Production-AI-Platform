"""Tests for the tenant invoice-preflight application service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from langchain_core.documents import Document

from rag.billing_requirements import BillingRequirements
from rag.billing_requirements_service import (
    BillingRequirementsServiceError,
)
from rag.invoice_facts import (
    InvoiceFacts,
    InvoiceFactsExtractionError,
)
from rag.invoice_preflight import (
    FindingSeverity,
    PaymentReadiness,
    PreflightField,
)
from rag.invoice_preflight_service import (
    InvoicePreflightService,
    InvoicePreflightServiceError,
)
from rag.models import (
    Document as DocumentRecord,
    DocumentType,
)
from rag.tenant_document_loader import TenantDocumentLoadError


ORGANIZATION_A = UUID(
    "10000000-0000-0000-0000-000000000001"
)
ORGANIZATION_B = UUID(
    "20000000-0000-0000-0000-000000000002"
)


class FakeBillingRequirementsService:
    """Controllable billing-requirements service."""

    def __init__(
        self,
        result: BillingRequirements | None = None,
        *,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.events = events
        self.calls: list[list[DocumentRecord]] = []

    def extract(
        self,
        documents: list[DocumentRecord],
    ) -> BillingRequirements:
        self.calls.append(documents)

        if self.events is not None:
            self.events.append("billing_requirements")

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError(
                "Fake billing requirements result was not configured."
            )

        return self.result


class FakeTenantDocumentLoader:
    """Controllable loader for the invoice document."""

    def __init__(
        self,
        result: list[Document] | None = None,
        *,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.events = events
        self.calls: list[DocumentRecord] = []

    def load(
        self,
        document: DocumentRecord,
    ) -> list[Document]:
        self.calls.append(document)

        if self.events is not None:
            self.events.append("invoice_load")

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError(
                "Fake invoice document result was not configured."
            )

        return self.result


class FakeInvoiceFactsExtractor:
    """Controllable grounded invoice-facts extractor."""

    def __init__(
        self,
        result: InvoiceFacts | None = None,
        *,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.events = events
        self.calls: list[list[Document]] = []

    def extract(
        self,
        documents: list[Document],
    ) -> InvoiceFacts:
        self.calls.append(documents)

        if self.events is not None:
            self.events.append("invoice_facts")

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError(
                "Fake invoice facts result was not configured."
            )

        return self.result


def _record(
    *,
    organization_id: UUID = ORGANIZATION_A,
    document_type: str,
    original_filename: str,
) -> DocumentRecord:
    record = SimpleNamespace(
        id=uuid4(),
        organization_id=organization_id,
        document_type=document_type,
        original_filename=original_filename,
        storage_key=f"test/{uuid4()}.md",
    )

    return cast(DocumentRecord, record)


def _requirements(**overrides: Any) -> BillingRequirements:
    data: dict[str, Any] = {
        "po_required": None,
        "po_number": None,
        "payment_terms": None,
        "milestone_approval_required": None,
        "billing_entity": None,
        "project_code": None,
        "required_attachments": None,
        "evidence": [],
    }
    data.update(overrides)
    return BillingRequirements.model_validate(data)


def _invoice_facts(**overrides: Any) -> InvoiceFacts:
    data: dict[str, Any] = {
        "invoice_number": None,
        "po_number": None,
        "payment_terms": None,
        "billing_entity": None,
        "project_code": None,
        "attachments": None,
        "evidence": [],
    }
    data.update(overrides)
    return InvoiceFacts.model_validate(data)


def _create_service(
    billing_service: FakeBillingRequirementsService,
    document_loader: FakeTenantDocumentLoader,
    invoice_extractor: FakeInvoiceFactsExtractor,
) -> InvoicePreflightService:
    return InvoicePreflightService(
        billing_requirements_service=cast(
            Any,
            billing_service,
        ),
        document_loader=cast(
            Any,
            document_loader,
        ),
        invoice_extractor=cast(
            Any,
            invoice_extractor,
        ),
    )


def test_evaluate_orchestrates_preflight_in_correct_order() -> None:
    events: list[str] = []
    billing_documents = [
        _record(
            document_type=DocumentType.CONTRACT.value,
            original_filename="contract.md",
        ),
        _record(
            document_type=(
                DocumentType.BILLING_INSTRUCTIONS.value
            ),
            original_filename="billing-instructions.md",
        ),
    ]
    invoice_document = _record(
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    loaded_invoice = [
        Document(
            page_content=(
                "Invoice Number: INV-1042\n"
                "Purchase Order: PO-4821\n"
                "Payment Terms: Net 45"
            ),
            metadata={
                "file_name": "invoice.md",
            },
        )
    ]
    billing_service = FakeBillingRequirementsService(
        _requirements(
            po_required=True,
            po_number="PO-4821",
            payment_terms="Net 45",
        ),
        events=events,
    )
    document_loader = FakeTenantDocumentLoader(
        loaded_invoice,
        events=events,
    )
    invoice_extractor = FakeInvoiceFactsExtractor(
        _invoice_facts(
            invoice_number="INV-1042",
            po_number="PO-4821",
            payment_terms="Net 45",
        ),
        events=events,
    )
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    result = service.evaluate(
        billing_documents,
        invoice_document,
    )

    assert result.payment_readiness is PaymentReadiness.READY
    assert events == [
        "billing_requirements",
        "invoice_load",
        "invoice_facts",
    ]
    assert billing_service.calls == [billing_documents]
    assert document_loader.calls == [invoice_document]
    assert invoice_extractor.calls == [loaded_invoice]
    assert all(
        finding.severity is FindingSeverity.PASS
        for finding in result.findings
    )


def test_evaluate_returns_deterministic_blocked_result() -> None:
    billing_documents = [
        _record(
            document_type=DocumentType.SOW.value,
            original_filename="sow.md",
        )
    ]
    invoice_document = _record(
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    billing_service = FakeBillingRequirementsService(
        _requirements(
            project_code="AI-2026-17",
        )
    )
    document_loader = FakeTenantDocumentLoader(
        [
            Document(
                page_content="Invoice Number: INV-1042",
                metadata={"file_name": "invoice.md"},
            )
        ]
    )
    invoice_extractor = FakeInvoiceFactsExtractor(
        _invoice_facts(
            invoice_number="INV-1042",
            project_code=None,
        )
    )
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    result = service.evaluate(
        billing_documents,
        invoice_document,
    )

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert len(result.findings) == 1
    assert (
        result.findings[0].field
        is PreflightField.PROJECT_CODE
    )
    assert (
        result.findings[0].severity
        is FindingSeverity.BLOCKER
    )


def test_empty_billing_documents_are_rejected_before_loading() -> None:
    invoice_document = _record(
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    billing_service = FakeBillingRequirementsService()
    document_loader = FakeTenantDocumentLoader()
    invoice_extractor = FakeInvoiceFactsExtractor()
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    with pytest.raises(
        InvoicePreflightServiceError,
        match="No billing requirement documents",
    ):
        service.evaluate(
            [],
            invoice_document,
        )

    assert billing_service.calls == []
    assert document_loader.calls == []
    assert invoice_extractor.calls == []


def test_non_invoice_document_is_rejected_before_loading() -> None:
    billing_documents = [
        _record(
            document_type=DocumentType.CONTRACT.value,
            original_filename="contract.md",
        )
    ]
    non_invoice_document = _record(
        document_type=DocumentType.SOW.value,
        original_filename="not-an-invoice.md",
    )
    billing_service = FakeBillingRequirementsService()
    document_loader = FakeTenantDocumentLoader()
    invoice_extractor = FakeInvoiceFactsExtractor()
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    with pytest.raises(
        InvoicePreflightServiceError,
        match="must have document type invoice",
    ):
        service.evaluate(
            billing_documents,
            non_invoice_document,
        )

    assert billing_service.calls == []
    assert document_loader.calls == []
    assert invoice_extractor.calls == []


def test_cross_tenant_invoice_is_rejected_before_loading() -> None:
    billing_documents = [
        _record(
            organization_id=ORGANIZATION_A,
            document_type=DocumentType.CONTRACT.value,
            original_filename="contract.md",
        )
    ]
    invoice_document = _record(
        organization_id=ORGANIZATION_B,
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    billing_service = FakeBillingRequirementsService()
    document_loader = FakeTenantDocumentLoader()
    invoice_extractor = FakeInvoiceFactsExtractor()
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    with pytest.raises(
        InvoicePreflightServiceError,
        match="must belong to the same organization",
    ):
        service.evaluate(
            billing_documents,
            invoice_document,
        )

    assert billing_service.calls == []
    assert document_loader.calls == []
    assert invoice_extractor.calls == []


def test_mixed_tenant_billing_documents_are_rejected() -> None:
    billing_documents = [
        _record(
            organization_id=ORGANIZATION_A,
            document_type=DocumentType.CONTRACT.value,
            original_filename="contract.md",
        ),
        _record(
            organization_id=ORGANIZATION_B,
            document_type=DocumentType.SOW.value,
            original_filename="sow.md",
        ),
    ]
    invoice_document = _record(
        organization_id=ORGANIZATION_A,
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    billing_service = FakeBillingRequirementsService()
    document_loader = FakeTenantDocumentLoader()
    invoice_extractor = FakeInvoiceFactsExtractor()
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    with pytest.raises(
        InvoicePreflightServiceError,
        match="must belong to the same organization",
    ):
        service.evaluate(
            billing_documents,
            invoice_document,
        )

    assert billing_service.calls == []
    assert document_loader.calls == []
    assert invoice_extractor.calls == []


def test_billing_requirements_error_is_preserved() -> None:
    billing_documents = [
        _record(
            document_type=DocumentType.SUPPORTING_EVIDENCE.value,
            original_filename="evidence.md",
        )
    ]
    invoice_document = _record(
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    expected_error = BillingRequirementsServiceError(
        "Unsupported billing requirement document type."
    )
    billing_service = FakeBillingRequirementsService(
        error=expected_error
    )
    document_loader = FakeTenantDocumentLoader()
    invoice_extractor = FakeInvoiceFactsExtractor()
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    with pytest.raises(
        BillingRequirementsServiceError,
    ) as error_info:
        service.evaluate(
            billing_documents,
            invoice_document,
        )

    assert error_info.value is expected_error
    assert billing_service.calls == [billing_documents]
    assert document_loader.calls == []
    assert invoice_extractor.calls == []


def test_invoice_loader_error_is_preserved() -> None:
    billing_documents = [
        _record(
            document_type=DocumentType.CONTRACT.value,
            original_filename="contract.md",
        )
    ]
    invoice_document = _record(
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    expected_error = TenantDocumentLoadError(
        "Stored document file was not found."
    )
    billing_service = FakeBillingRequirementsService(
        _requirements(po_required=True)
    )
    document_loader = FakeTenantDocumentLoader(
        error=expected_error
    )
    invoice_extractor = FakeInvoiceFactsExtractor()
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    with pytest.raises(
        TenantDocumentLoadError,
    ) as error_info:
        service.evaluate(
            billing_documents,
            invoice_document,
        )

    assert error_info.value is expected_error
    assert billing_service.calls == [billing_documents]
    assert document_loader.calls == [invoice_document]
    assert invoice_extractor.calls == []


def test_invoice_extraction_error_is_preserved() -> None:
    billing_documents = [
        _record(
            document_type=DocumentType.CONTRACT.value,
            original_filename="contract.md",
        )
    ]
    invoice_document = _record(
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )
    loaded_invoice = [
        Document(
            page_content="Invoice Number: INV-1042",
            metadata={"file_name": "invoice.md"},
        )
    ]
    expected_error = InvoiceFactsExtractionError(
        "The invoice extraction response was invalid."
    )
    billing_service = FakeBillingRequirementsService(
        _requirements(po_required=True)
    )
    document_loader = FakeTenantDocumentLoader(
        loaded_invoice
    )
    invoice_extractor = FakeInvoiceFactsExtractor(
        error=expected_error
    )
    service = _create_service(
        billing_service,
        document_loader,
        invoice_extractor,
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
    ) as error_info:
        service.evaluate(
            billing_documents,
            invoice_document,
        )

    assert error_info.value is expected_error
    assert billing_service.calls == [billing_documents]
    assert document_loader.calls == [invoice_document]
    assert invoice_extractor.calls == [loaded_invoice]
