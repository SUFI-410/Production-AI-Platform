from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import api.billing_requirement_routes as route_module
from api.dependencies import (
    get_current_organization,
    get_db,
)
from api.main import app
from rag.billing_requirements import BillingRequirements
from rag.billing_requirements_service import (
    BillingRequirementsService,
    BillingRequirementsServiceError,
)
from rag.models import (
    Document as DocumentRecord,
    DocumentType,
    Organization,
)


ORGANIZATION_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)

OTHER_ORGANIZATION_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)

DOCUMENT_ID_1 = UUID(
    "33333333-3333-3333-3333-333333333333"
)

DOCUMENT_ID_2 = UUID(
    "44444444-4444-4444-4444-444444444444"
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
        self.statements.append(
            statement
        )

        return FakeScalarResult(
            self.records
        )


class FakeBillingRequirementsService:
    def __init__(
        self,
        result: BillingRequirements | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = (
            result
            or BillingRequirements()
        )

        self.error = error

        self.calls: list[
            list[DocumentRecord]
        ] = []

    def extract(
        self,
        documents: list[DocumentRecord],
    ) -> BillingRequirements:
        self.calls.append(
            documents
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
    document_type: str = (
        DocumentType.CONTRACT.value
    ),
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        organization_id=organization_id,
        uploaded_by_user_id=None,
        original_filename="contract.md",
        content_type="text/markdown",
        size_bytes=100,
        storage_key=(
            f"{organization_id}/"
            f"{document_id}.md"
        ),
        document_type=document_type,
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
    service: FakeBillingRequirementsService,
) -> None:
    app.dependency_overrides[
        route_module.get_billing_requirements_service
    ] = lambda: cast(
        BillingRequirementsService,
        service,
    )


def test_extract_returns_billing_requirements() -> None:
    first = _make_document(
        DOCUMENT_ID_1
    )

    second = _make_document(
        DOCUMENT_ID_2,
        document_type=(
            DocumentType.SOW.value
        ),
    )

    db = FakeSession(
        [
            second,
            first,
        ]
    )

    service = FakeBillingRequirementsService(
        result=BillingRequirements(
            po_required=True,
            po_number="PO-4821",
            payment_terms="Net 45",
        )
    )

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/billing-requirements/extract",
        json={
            "document_ids": [
                str(DOCUMENT_ID_1),
                str(DOCUMENT_ID_2),
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["po_required"] is True
    assert body["po_number"] == "PO-4821"
    assert body["payment_terms"] == "Net 45"

    assert len(service.calls) == 1

    assert [
        document.id
        for document in service.calls[0]
    ] == [
        DOCUMENT_ID_1,
        DOCUMENT_ID_2,
    ]

    assert len(db.statements) == 1

    compiled = db.statements[
        0
    ].compile()

    sql = str(
        compiled
    )

    assert (
        "documents.organization_id"
        in sql
    )

    assert (
        ORGANIZATION_ID
        in compiled.params.values()
    )


def test_extract_rejects_missing_or_foreign_document(
) -> None:
    first = _make_document(
        DOCUMENT_ID_1
    )

    db = FakeSession(
        [first]
    )

    service = FakeBillingRequirementsService()

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/billing-requirements/extract",
        json={
            "document_ids": [
                str(DOCUMENT_ID_1),
                str(DOCUMENT_ID_2),
            ]
        },
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "One or more documents were not found."
        )
    }

    assert service.calls == []


def test_extract_rejects_duplicate_document_ids(
) -> None:
    db = FakeSession(
        []
    )

    service = FakeBillingRequirementsService()

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/billing-requirements/extract",
        json={
            "document_ids": [
                str(DOCUMENT_ID_1),
                str(DOCUMENT_ID_1),
            ]
        },
    )

    assert response.status_code == 422

    assert service.calls == []


def test_extract_rejects_client_organization_id(
) -> None:
    db = FakeSession(
        []
    )

    service = FakeBillingRequirementsService()

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/billing-requirements/extract",
        json={
            "document_ids": [
                str(DOCUMENT_ID_1),
            ],
            "organization_id": str(
                OTHER_ORGANIZATION_ID
            ),
        },
    )

    assert response.status_code == 422

    assert service.calls == []


def test_extract_maps_service_validation_error(
) -> None:
    record = _make_document(
        DOCUMENT_ID_1
    )

    db = FakeSession(
        [record]
    )

    service = FakeBillingRequirementsService(
        error=BillingRequirementsServiceError(
            "Unsupported billing requirement "
            "document type(s): invoice"
        )
    )

    _override_database(db)
    _override_organization()
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/billing-requirements/extract",
        json={
            "document_ids": [
                str(DOCUMENT_ID_1),
            ]
        },
    )

    assert response.status_code == 422

    assert response.json() == {
        "detail": (
            "Unsupported billing requirement "
            "document type(s): invoice"
        )
    }


def test_extract_requires_authentication() -> None:
    db = FakeSession(
        []
    )

    service = FakeBillingRequirementsService()

    _override_database(db)
    _override_service(service)

    client = TestClient(app)

    response = client.post(
        "/billing-requirements/extract",
        json={
            "document_ids": [
                str(DOCUMENT_ID_1),
            ]
        },
    )

    assert response.status_code == 401

    assert service.calls == []
