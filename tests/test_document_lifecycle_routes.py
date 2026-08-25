from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import api.document_routes as document_routes_module
from api.dependencies import (
    get_current_organization,
    get_current_user,
    get_db,
)
from api.main import app
from rag.document_storage import LocalDocumentStorage
from rag.models import (
    Document as DocumentRecord,
    DocumentStatus,
    DocumentType,
    Organization,
    User,
)


ORGANIZATION_ID = UUID("10000000-0000-0000-0000-000000000001")
OTHER_ORGANIZATION_ID = UUID("20000000-0000-0000-0000-000000000002")
USER_ID = UUID("30000000-0000-0000-0000-000000000003")
DOCUMENT_ID = UUID("40000000-0000-0000-0000-000000000004")
CREATED_AT = datetime(
    2026,
    8,
    25,
    10,
    0,
    tzinfo=UTC,
)


class FakeScalarResult:
    def __init__(
        self,
        records: list[DocumentRecord],
    ) -> None:
        self.records = records

    def all(self) -> list[DocumentRecord]:
        return self.records


class FakeSession:
    def __init__(
        self,
        *,
        records: list[DocumentRecord] | None = None,
        selected_document: DocumentRecord | None = None,
        selection_error: Exception | None = None,
        commit_error: Exception | None = None,
    ) -> None:
        self.records = records or []
        self.selected_document = selected_document
        self.selection_error = selection_error
        self.commit_error = commit_error

        self.statements: list[Any] = []
        self.deleted: list[DocumentRecord] = []
        self.commit_called = False
        self.rollback_called = False

    def scalars(
        self,
        statement: Any,
    ) -> FakeScalarResult:
        self.statements.append(statement)

        if self.selection_error is not None:
            raise self.selection_error

        return FakeScalarResult(self.records)

    def scalar(
        self,
        statement: Any,
    ) -> DocumentRecord | None:
        self.statements.append(statement)

        if self.selection_error is not None:
            raise self.selection_error

        return self.selected_document

    def delete(
        self,
        document: DocumentRecord,
    ) -> None:
        self.deleted.append(document)

    def commit(self) -> None:
        self.commit_called = True

        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self) -> None:
        self.rollback_called = True


@pytest.fixture
def document_storage(
    tmp_path: Path,
) -> LocalDocumentStorage:
    return LocalDocumentStorage(tmp_path / "uploads")


@pytest.fixture(autouse=True)
def clear_dependency_overrides(
    document_storage: LocalDocumentStorage,
) -> Iterator[None]:
    app.dependency_overrides.clear()
    app.dependency_overrides[document_routes_module.get_document_storage] = lambda: (
        document_storage
    )

    yield

    app.dependency_overrides.clear()


def _make_user() -> User:
    return User(
        id=USER_ID,
        organization_id=ORGANIZATION_ID,
        email="owner@example.com",
        password_hash="stored-password-hash",
        is_active=True,
    )


def _make_organization() -> Organization:
    return Organization(
        id=ORGANIZATION_ID,
        name="Acme AI",
    )


def _make_document(
    *,
    organization_id: UUID = ORGANIZATION_ID,
    storage_key: str | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        id=DOCUMENT_ID,
        organization_id=organization_id,
        uploaded_by_user_id=USER_ID,
        original_filename="invoice.md",
        content_type="text/markdown",
        size_bytes=17,
        storage_key=storage_key,
        document_type=DocumentType.INVOICE.value,
        status=DocumentStatus.UPLOADED.value,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _override_database(
    db: FakeSession,
) -> None:
    app.dependency_overrides[get_db] = lambda: cast(
        Session,
        db,
    )


def _override_authenticated_tenant() -> None:
    app.dependency_overrides[get_current_user] = lambda: _make_user()
    app.dependency_overrides[get_current_organization] = lambda: _make_organization()


def _statement_parameter_values(
    statement: Any,
) -> set[Any]:
    return set(statement.compile().params.values())


def test_list_documents_returns_tenant_documents() -> None:
    document = _make_document()
    db = FakeSession(records=[document])

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).get("/documents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(DOCUMENT_ID),
            "organization_id": str(ORGANIZATION_ID),
            "uploaded_by_user_id": str(USER_ID),
            "original_filename": "invoice.md",
            "content_type": "text/markdown",
            "size_bytes": 17,
            "document_type": DocumentType.INVOICE.value,
            "status": DocumentStatus.UPLOADED.value,
            "created_at": CREATED_AT.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "updated_at": CREATED_AT.isoformat().replace(
                "+00:00",
                "Z",
            ),
        }
    ]
    assert ORGANIZATION_ID in (_statement_parameter_values(db.statements[0]))


def test_list_documents_requires_authentication() -> None:
    response = TestClient(app).get("/documents")

    assert response.status_code == 401


def test_list_documents_handles_database_failure() -> None:
    db = FakeSession(selection_error=SQLAlchemyError("database failure"))

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).get("/documents")

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to list documents."}


def test_delete_document_removes_file_and_record(
    document_storage: LocalDocumentStorage,
) -> None:
    storage_key = f"{ORGANIZATION_ID}/{DOCUMENT_ID}.md"
    document_storage.save(
        storage_key,
        b"# Private invoice",
    )
    document = _make_document(storage_key=storage_key)
    db = FakeSession(selected_document=document)

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).delete(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 204
    assert response.content == b""
    assert not document_storage.path_for(storage_key).exists()
    assert db.deleted == [document]
    assert db.commit_called is True
    assert db.rollback_called is False
    assert ORGANIZATION_ID in (_statement_parameter_values(db.statements[0]))


def test_delete_document_hides_other_tenant() -> None:
    db = FakeSession(selected_document=None)

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).delete(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    assert db.deleted == []
    assert db.commit_called is False
    assert ORGANIZATION_ID in (_statement_parameter_values(db.statements[0]))


def test_delete_document_rejects_invalid_id() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).delete("/documents/not-a-uuid")

    assert response.status_code == 422
    assert db.statements == []


def test_delete_document_handles_storage_failure() -> None:
    document = _make_document(storage_key="../../outside.md")
    db = FakeSession(selected_document=document)

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).delete(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to delete the document."}
    assert db.deleted == []
    assert db.commit_called is False


def test_delete_document_rolls_back_database_failure(
    document_storage: LocalDocumentStorage,
) -> None:
    storage_key = f"{ORGANIZATION_ID}/{DOCUMENT_ID}.md"
    document_storage.save(
        storage_key,
        b"# Private invoice",
    )
    document = _make_document(storage_key=storage_key)
    db = FakeSession(
        selected_document=document,
        commit_error=SQLAlchemyError("database failure"),
    )

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).delete(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to delete the document."}
    assert db.deleted == [document]
    assert db.commit_called is True
    assert db.rollback_called is True
    assert not document_storage.path_for(storage_key).exists()


def test_document_lifecycle_openapi_requires_bearer_auth() -> None:
    schema = app.openapi()

    for method in ("get", "delete"):
        path = "/documents" if method == "get" else "/documents/{document_id}"
        security = schema["paths"][path][method]["security"]

        assert {"HTTPBearer": []} in security
