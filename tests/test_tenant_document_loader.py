from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from rag.document_storage import LocalDocumentStorage
from rag.models import (
    Document as DocumentRecord,
    DocumentType,
)
from rag.tenant_document_loader import (
    TenantDocumentLoadError,
    TenantDocumentLoader,
)


ORGANIZATION_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)

OTHER_ORGANIZATION_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)

DOCUMENT_ID = UUID(
    "33333333-3333-3333-3333-333333333333"
)


def _make_document(
    *,
    organization_id: UUID = ORGANIZATION_ID,
    storage_key: str | None = (
        f"{ORGANIZATION_ID}/{DOCUMENT_ID}.md"
    ),
) -> DocumentRecord:
    return DocumentRecord(
        id=DOCUMENT_ID,
        organization_id=organization_id,
        uploaded_by_user_id=None,
        original_filename="contract.md",
        content_type="text/markdown",
        size_bytes=25,
        storage_key=storage_key,
        document_type=DocumentType.CONTRACT.value,
    )


def test_load_markdown_enriches_tenant_metadata(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(
        tmp_path / "uploads"
    )

    record = _make_document()

    assert record.storage_key is not None

    storage.save(
        record.storage_key,
        (
            b"# Contract\n\n"
            b"Payment terms are Net 45."
        ),
    )

    loader = TenantDocumentLoader(
        storage
    )

    documents = loader.load(
        record
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        "Payment terms are Net 45."
        in document.page_content
    )

    assert (
        document.metadata[
            "document_type"
        ]
        == "markdown"
    )

    assert (
        document.metadata[
            "business_document_type"
        ]
        == DocumentType.CONTRACT.value
    )

    assert (
        document.metadata[
            "tenant_document_id"
        ]
        == str(DOCUMENT_ID)
    )

    assert (
        document.metadata[
            "organization_id"
        ]
        == str(ORGANIZATION_ID)
    )

    assert (
        document.metadata[
            "original_filename"
        ]
        == "contract.md"
    )


def test_load_rejects_missing_storage_key(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(
        tmp_path / "uploads"
    )

    loader = TenantDocumentLoader(
        storage
    )

    record = _make_document(
        storage_key=None
    )

    with pytest.raises(
        TenantDocumentLoadError,
        match="no durable storage key",
    ):
        loader.load(
            record
        )


def test_load_rejects_missing_stored_file(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(
        tmp_path / "uploads"
    )

    loader = TenantDocumentLoader(
        storage
    )

    record = _make_document()

    with pytest.raises(
        TenantDocumentLoadError,
        match="file was not found",
    ):
        loader.load(
            record
        )


def test_load_many_rejects_cross_tenant_documents(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(
        tmp_path / "uploads"
    )

    loader = TenantDocumentLoader(
        storage
    )

    first = _make_document()

    second = _make_document(
        organization_id=OTHER_ORGANIZATION_ID,
        storage_key=(
            f"{OTHER_ORGANIZATION_ID}/"
            f"{DOCUMENT_ID}.md"
        ),
    )

    with pytest.raises(
        TenantDocumentLoadError,
        match="multiple organizations",
    ):
        loader.load_many(
            [
                first,
                second,
            ]
        )
