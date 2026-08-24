"""Tests for the authenticated invoice-preflight API route."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import api.invoice_preflight_routes as route_module
from api.dependencies import (
    get_current_organization,
    get_db,
)
from api.main import app
from rag.billing_requirements import (
    BillingRequirementsExtractionError,
)
from rag.billing_requirements_service import (
    BillingRequirementsServiceError,
)
from rag.invoice_facts import InvoiceFactsExtractionError
from rag.invoice_preflight import (
    FindingSeverity,
    InvoicePreflightResult,
    PaymentReadiness,
    PreflightField,
    PreflightFinding,
)
from rag.invoice_preflight_service import (
    InvoicePreflightService,
    InvoicePreflightServiceError,
)
from rag.models import (
    Document as DocumentRecord,
    DocumentType,
    Organization,
)
from rag.tenant_document_loader import TenantDocumentLoadError


ORGANIZATION_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)

OTHER_ORGANIZATION_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)

BILLING_DOCUMENT_ID_1 = UUID(
    "33333333-3333-3333-3333-333333333333"
)

BILLING_DOCUMENT_ID_2 = UUID(
    "44444444-4444-4444-4444-444444444444"
)

INVOICE_DOCUMENT_ID = UUID(
    "55555555-5555-5555-5555-555555555555"
)


class FakeScalarResult:
    def __init__(
        self,
        records: list[DocumentRecord],
    ) -> None:
        self.records = records

    def all(
        self,
    ) -> list[DocumentRecord]:
        return self.records


class FakeSession:
    def __init__(
        self,
        records: list[DocumentRecord],
    ) -> None:
        self.records = records
        self.statements: list[Any] = []

    def scalars(
        self,
        statement: Any,
    ) -> FakeScalarResult:
        self.statements.append(statement)

        return FakeScalarResult(
            self.records
        )


class FakeInvoicePreflightService:
    def __init__(
        self,
        result: InvoicePreflightResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = (
            result
            or InvoicePreflightResult(
                payment_readiness=PaymentReadiness.READY,
                findings=[],
            )
        )
        self.error = error
        self.calls: list[
            tuple[
                list[DocumentRecord],
                DocumentRecord,
            ]
        ] = []

    def evaluate(
        self,
        billing_documents: list[DocumentRecord],
        invoice_document: DocumentRecord,
    ) -> InvoicePreflightResult:
        self.calls.append(
            (
                billing_documents,
                invoice_document,
            )
        )

        if self.error is not None:
            raise self.error

        return self.result


@pytest.fixture(autouse=True)
def clear_dependency_overrides(
) -> Iterator[None]:
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


def _make_organization() -> Organization:
    return Organization(
        id=ORGANIZATION_ID,
        name="Acme AI",
    )


def _make_document(
    document_id: UUID,
    *,
    organization_id: UUID = ORGANIZATION_ID,
    document_type: str,
    original_filename: str,
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        organization_id=organization_id,
        uploaded_by_user_id=None,
        original_filename=original_filename,
        content_type="text/markdown",
        size_bytes=100,
        storage_key=(
            f"{organization_id}/"
            f"{document_id}.md"
        ),
        document_type=document_type,
    )


def _billing_document_one() -> DocumentRecord:
    return _make_document(
        BILLING_DOCUMENT_ID_1,
        document_type=DocumentType.CONTRACT.value,
        original_filename="contract.md",
    )


def _billing_document_two() -> DocumentRecord:
    return _make_document(
        BILLING_DOCUMENT_ID_2,
        document_type=(
            DocumentType.BILLING_INSTRUCTIONS.value
        ),
        original_filename="billing-instructions.md",
    )


def _invoice_document() -> DocumentRecord:
    return _make_document(
        INVOICE_DOCUMENT_ID,
        document_type=DocumentType.INVOICE.value,
        original_filename="invoice.md",
    )


def _override_database(
    db: FakeSession,
) -> None:
    app.dependency_overrides[
        get_db
    ] = lambda: cast(
        Session,
        db,
    )


def _override_organization() -> None:
    organization = _make_organization()

    app.dependency_overrides[
        get_current_organization
    ] = lambda: organization


def _override_service(
    service: FakeInvoicePreflightService,
) -> None:
    app.dependency_overrides[
        route_module.get_invoice_preflight_service
    ] = lambda: cast(
        InvoicePreflightService,
        service,
    )


def test_evaluate_returns_preflight_result_in_requested_order(
) -> None:
    first = _billing_document_one()
    second = _billing_document_two()
    invoice = _invoice_document()

    db = FakeSession(
        [
            invoice,
            second,
            first,
        ]
    )

    service = FakeInvoicePreflightService(
        result=InvoicePreflightResult(
            payment_readiness=PaymentReadiness.BLOCKED,
            findings=[
                PreflightFinding(
                    severity=FindingSeverity.BLOCKER,
                    field=PreflightField.PROJECT_CODE,
                    message=(
                        "Invoice is missing required project "
                        "code AI-2026-17."
                    ),
                ),
                PreflightFinding(
                    severity=FindingSeverity.PASS,
                    field=PreflightField.PAYMENT_TERMS,
                    message=(
                        "Invoice correctly uses required "
                        "payment terms Net 45."
                    ),
                ),
            ],
        )
    )

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
                str(BILLING_DOCUMENT_ID_2),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "payment_readiness": "BLOCKED",
        "findings": [
            {
                "severity": "BLOCKER",
                "field": "project_code",
                "message": (
                    "Invoice is missing required project "
                    "code AI-2026-17."
                ),
            },
            {
                "severity": "PASS",
                "field": "payment_terms",
                "message": (
                    "Invoice correctly uses required "
                    "payment terms Net 45."
                ),
            },
        ],
    }

    assert len(service.calls) == 1

    billing_documents, invoice_document = service.calls[0]

    assert [
        document.id
        for document in billing_documents
    ] == [
        BILLING_DOCUMENT_ID_1,
        BILLING_DOCUMENT_ID_2,
    ]

    assert invoice_document.id == INVOICE_DOCUMENT_ID

    assert len(db.statements) == 1

    compiled = db.statements[0].compile()
    sql = str(compiled)

    assert "documents.organization_id" in sql
    assert ORGANIZATION_ID in compiled.params.values()


def test_evaluate_rejects_missing_or_foreign_document() -> None:
    first = _billing_document_one()
    invoice = _invoice_document()

    db = FakeSession(
        [
            first,
            invoice,
        ]
    )

    service = FakeInvoicePreflightService()

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
                str(BILLING_DOCUMENT_ID_2),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "One or more documents were not found."
        )
    }

    assert service.calls == []


def test_evaluate_rejects_duplicate_billing_document_ids(
) -> None:
    db = FakeSession([])
    service = FakeInvoicePreflightService()

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
                str(BILLING_DOCUMENT_ID_1),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 422
    assert db.statements == []
    assert service.calls == []


def test_evaluate_rejects_invoice_in_billing_document_ids(
) -> None:
    db = FakeSession([])
    service = FakeInvoicePreflightService()

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
                str(INVOICE_DOCUMENT_ID),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 422
    assert db.statements == []
    assert service.calls == []


def test_evaluate_rejects_client_organization_id() -> None:
    db = FakeSession([])
    service = FakeInvoicePreflightService()

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
            "organization_id": str(
                OTHER_ORGANIZATION_ID
            ),
        },
    )

    assert response.status_code == 422
    assert db.statements == []
    assert service.calls == []


@pytest.mark.parametrize(
    "error",
    [
        InvoicePreflightServiceError(
            "The invoice document must have "
            "document type invoice."
        ),
        BillingRequirementsServiceError(
            "Unsupported billing requirement "
            "document type(s): supporting_evidence"
        ),
    ],
)
def test_evaluate_maps_service_validation_errors(
    error: Exception,
) -> None:
    first = _billing_document_one()
    invoice = _invoice_document()

    db = FakeSession(
        [
            first,
            invoice,
        ]
    )

    service = FakeInvoicePreflightService(
        error=error
    )

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": str(error)
    }


def test_evaluate_maps_document_load_error() -> None:
    first = _billing_document_one()
    invoice = _invoice_document()

    db = FakeSession(
        [
            first,
            invoice,
        ]
    )

    service = FakeInvoicePreflightService(
        error=TenantDocumentLoadError(
            "Stored document file was not found."
        )
    )

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 409

    assert response.json() == {
        "detail": (
            "One or more documents are not "
            "available for processing."
        )
    }


@pytest.mark.parametrize(
    "error",
    [
        BillingRequirementsExtractionError(
            "Billing extraction failed."
        ),
        InvoiceFactsExtractionError(
            "Invoice extraction failed."
        ),
    ],
)
def test_evaluate_maps_extraction_errors(
    error: Exception,
) -> None:
    first = _billing_document_one()
    invoice = _invoice_document()

    db = FakeSession(
        [
            first,
            invoice,
        ]
    )

    service = FakeInvoicePreflightService(
        error=error
    )

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 502

    assert response.json() == {
        "detail": (
            "Unable to extract invoice preflight data "
            "from the documents."
        )
    }


def test_evaluate_requires_authentication() -> None:
    db = FakeSession([])
    service = FakeInvoicePreflightService()

    _override_database(db)
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/invoice-preflight/evaluate",
        json={
            "billing_document_ids": [
                str(BILLING_DOCUMENT_ID_1),
            ],
            "invoice_document_id": str(
                INVOICE_DOCUMENT_ID
            ),
        },
    )

    assert response.status_code == 401
    assert service.calls == []
