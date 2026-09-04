from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi import HTTPException
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
from rag.billing_service import (
    DocumentUploadNotAllowedError,
)
from rag.document_storage import LocalDocumentStorage
from rag.models import (
    Document as DocumentRecord,
    DocumentStatus,
    DocumentType,
    Organization,
    User,
)


USER_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)

ORGANIZATION_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)

OTHER_ORGANIZATION_ID = UUID(
    "33333333-3333-3333-3333-333333333333"
)

DOCUMENT_ID = UUID(
    "44444444-4444-4444-4444-444444444444"
)

CREATED_AT = datetime(
    2026,
    8,
    12,
    12,
    0,
    tzinfo=UTC,
)


class FakeSession:
    def __init__(
        self,
        *,
        commit_error: Exception | None = None,
    ) -> None:
        self.commit_error = commit_error

        self.added: list[Any] = []
        self.commit_called = False
        self.rollback_called = False
        self.refresh_called = False

    def add(
        self,
        instance: Any,
    ) -> None:
        self.added.append(instance)

    def commit(self) -> None:
        self.commit_called = True

        if self.commit_error is not None:
            raise self.commit_error

    def refresh(
        self,
        instance: Any,
    ) -> None:
        self.refresh_called = True

        if isinstance(
            instance,
            DocumentRecord,
        ):
            instance.id = DOCUMENT_ID
            instance.created_at = CREATED_AT
            instance.updated_at = CREATED_AT

    def rollback(self) -> None:
        self.rollback_called = True


class FakeBillingService:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.organization_ids: list[UUID] = []

    def ensure_document_upload_allowed(
        self,
        organization_id: UUID,
    ) -> None:
        self.organization_ids.append(organization_id)

        if self.error is not None:
            raise self.error


@pytest.fixture
def document_storage(
    tmp_path: Path,
) -> LocalDocumentStorage:
    """
    Provide isolated document storage for every route test.
    """

    return LocalDocumentStorage(
        tmp_path / "uploads"
    )


@pytest.fixture
def billing_service() -> FakeBillingService:
    return FakeBillingService()


@pytest.fixture(autouse=True)
def clear_dependency_overrides(
    document_storage: LocalDocumentStorage,
    billing_service: FakeBillingService,
) -> Iterator[None]:
    app.dependency_overrides.clear()

    app.dependency_overrides[
        document_routes_module.get_document_storage
    ] = lambda: document_storage

    app.dependency_overrides[
        document_routes_module.get_billing_service
    ] = lambda: billing_service

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


def _override_database(
    db: FakeSession,
) -> None:
    app.dependency_overrides[
        get_db
    ] = lambda: cast(
        Session,
        db,
    )


def _override_authenticated_tenant() -> None:
    user = _make_user()
    organization = _make_organization()

    app.dependency_overrides[
        get_current_user
    ] = lambda: user

    app.dependency_overrides[
        get_current_organization
    ] = lambda: organization


def test_upload_markdown_creates_document_record() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    content = (
        b"# Private Guide\n\n"
        b"Tenant content."
    )

    response = client.post(
        "/documents",
        files={
            "file": (
                "guide.md",
                content,
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201

    assert response.json() == {
        "id": str(DOCUMENT_ID),
        "organization_id": str(
            ORGANIZATION_ID
        ),
        "uploaded_by_user_id": str(
            USER_ID
        ),
        "original_filename": "guide.md",
        "content_type": "text/markdown",
        "size_bytes": len(content),
        "document_type": (
            DocumentType.OTHER.value
        ),
        "status": (
            DocumentStatus.UPLOADED.value
        ),
        "created_at": (
            CREATED_AT.isoformat().replace(
                "+00:00",
                "Z",
            )
        ),
        "updated_at": (
            CREATED_AT.isoformat().replace(
                "+00:00",
                "Z",
            )
        ),
    }

    assert db.commit_called is True
    assert db.refresh_called is True
    assert db.rollback_called is False

    assert len(db.added) == 1

    document = db.added[0]

    assert isinstance(
        document,
        DocumentRecord,
    )

    assert (
        document.organization_id
        == ORGANIZATION_ID
    )

    assert (
        document.uploaded_by_user_id
        == USER_ID
    )

    assert (
        document.original_filename
        == "guide.md"
    )

    assert (
        document.content_type
        == "text/markdown"
    )

    assert (
        document.document_type
        == DocumentType.OTHER.value
    )

    assert document.storage_key is not None

    assert (
        str(ORGANIZATION_ID)
        in document.storage_key
    )


def test_upload_rejects_exhausted_document_allowance(
    document_storage: LocalDocumentStorage,
    billing_service: FakeBillingService,
) -> None:
    db = FakeSession()

    billing_service.error = DocumentUploadNotAllowedError(
        "Document allowance has been exhausted."
    )

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).post(
        "/documents",
        files={
            "file": (
                "guide.md",
                b"# Guide",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": (
            "Document upload is not allowed by "
            "the current billing entitlement."
        )
    }
    assert billing_service.organization_ids == [
        ORGANIZATION_ID
    ]
    assert db.added == []
    assert db.commit_called is False
    assert db.rollback_called is True

    stored_files = [
        path
        for path in document_storage.root_directory.rglob("*")
        if path.is_file()
    ]

    assert stored_files == []


def test_upload_pdf_creates_document_record() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    pdf_content = (
        b"%PDF-1.7\n"
        b"minimal test document"
    )

    response = client.post(
        "/documents",
        files={
            "file": (
                "report.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    assert (
        response.json()[
            "original_filename"
        ]
        == "report.pdf"
    )

    assert (
        response.json()[
            "content_type"
        ]
        == "application/pdf"
    )

    assert (
        response.json()[
            "size_bytes"
        ]
        == len(pdf_content)
    )

    assert (
        response.json()[
            "document_type"
        ]
        == DocumentType.OTHER.value
    )


def test_upload_accepts_document_type() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        data={
            "document_type": (
                DocumentType.CONTRACT.value
            ),
        },
        files={
            "file": (
                "master-service-agreement.md",
                (
                    b"# Master Service Agreement\n\n"
                    b"Net 45."
                ),
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201

    assert (
        response.json()[
            "document_type"
        ]
        == DocumentType.CONTRACT.value
    )

    assert len(db.added) == 1

    document = db.added[0]

    assert isinstance(
        document,
        DocumentRecord,
    )

    assert (
        document.document_type
        == DocumentType.CONTRACT.value
    )

    assert document.storage_key is not None


def test_upload_rejects_invalid_document_type() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        data={
            "document_type": "random_invalid_type",
        },
        files={
            "file": (
                "contract.md",
                b"# Contract",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 422

    assert db.added == []


def test_upload_uses_authenticated_organization() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        data={
            "organization_id": str(
                OTHER_ORGANIZATION_ID
            )
        },
        files={
            "file": (
                "tenant.md",
                b"# Tenant document",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201

    document = db.added[0]

    assert (
        document.organization_id
        == ORGANIZATION_ID
    )

    assert (
        document.organization_id
        != OTHER_ORGANIZATION_ID
    )


def test_upload_uses_authenticated_user_as_uploader() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "owner.md",
                b"# Owner document",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201

    document = db.added[0]

    assert (
        document.uploaded_by_user_id
        == USER_ID
    )


def test_upload_rejects_missing_authentication() -> None:
    db = FakeSession()

    _override_database(db)

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "private.md",
                b"# Private",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": (
            "Invalid or missing "
            "authentication credentials."
        )
    }

    assert db.added == []


def test_upload_rejects_invalid_organization() -> None:
    db = FakeSession()
    user = _make_user()

    _override_database(db)

    app.dependency_overrides[
        get_current_user
    ] = lambda: user

    def reject_organization() -> Organization:
        raise HTTPException(
            status_code=403,
            detail=(
                "Authenticated user is not "
                "assigned to a valid organization."
            ),
        )

    app.dependency_overrides[
        get_current_organization
    ] = reject_organization

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "private.md",
                b"# Private",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 403

    assert db.added == []


def test_upload_rejects_unsupported_extension() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "malware.exe",
                b"not allowed",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 415

    assert response.json() == {
        "detail": (
            "Only PDF and Markdown "
            "files are supported."
        )
    }

    assert db.added == []


def test_upload_rejects_wrong_content_type() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "guide.md",
                b"# Guide",
                "image/png",
            )
        },
    )

    assert response.status_code == 415

    assert response.json() == {
        "detail": (
            "The uploaded file has an "
            "unsupported content type."
        )
    }


def test_upload_rejects_empty_file() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "empty.md",
                b"",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "The uploaded file is empty."
        )
    }


def test_upload_rejects_invalid_pdf_signature() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "fake.pdf",
                b"This is not a PDF.",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "The uploaded file is not "
            "a valid PDF."
        )
    }


def test_upload_rejects_non_utf8_markdown() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "invalid.md",
                b"\xff\xfe\xfa",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "Markdown files must use "
            "UTF-8 encoding."
        )
    }


def test_upload_rejects_whitespace_only_markdown() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "blank.md",
                b"   \n\t   ",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "The uploaded Markdown "
            "file is empty."
        )
    }


def test_upload_rejects_oversized_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    monkeypatch.setattr(
        document_routes_module,
        "MAX_DOCUMENT_UPLOAD_BYTES",
        8,
    )

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "large.md",
                b"123456789",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 413

    assert response.json() == {
        "detail": (
            "The uploaded file exceeds "
            "the 10 MiB limit."
        )
    }

    assert db.added == []


def test_upload_sanitizes_filename_to_basename() -> None:
    db = FakeSession()

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "../../private/guide.md",
                b"# Safe filename",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201

    document = db.added[0]

    assert (
        document.original_filename
        == "guide.md"
    )

    assert (
        response.json()[
            "original_filename"
        ]
        == "guide.md"
    )


def test_upload_rolls_back_database_failure(
    document_storage: LocalDocumentStorage,
) -> None:
    db = FakeSession(
        commit_error=SQLAlchemyError(
            "database failure"
        )
    )

    _override_database(db)
    _override_authenticated_tenant()

    client = TestClient(app)

    response = client.post(
        "/documents",
        files={
            "file": (
                "guide.md",
                b"# Guide",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Unable to create the "
            "document record."
        )
    }

    assert db.commit_called is True
    assert db.rollback_called is True

    stored_files = [
        path
        for path in document_storage.root_directory.rglob("*")
        if path.is_file()
    ]

    assert stored_files == []


def test_upload_rolls_back_when_storage_fails(
    monkeypatch: pytest.MonkeyPatch,
    document_storage: LocalDocumentStorage,
) -> None:
    db = FakeSession()

    def fail_save(
        _storage_key: str,
        _content: bytes,
    ) -> Path:
        raise document_routes_module.DocumentStorageError(
            "storage failure"
        )

    monkeypatch.setattr(
        document_storage,
        "save",
        fail_save,
    )

    _override_database(db)
    _override_authenticated_tenant()

    response = TestClient(app).post(
        "/documents",
        files={
            "file": (
                "guide.md",
                b"# Guide",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Unable to store the document."
    }
    assert db.added == []
    assert db.commit_called is False
    assert db.rollback_called is True
