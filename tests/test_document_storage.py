from __future__ import annotations

from pathlib import Path

import pytest

from rag.document_storage import (
    DocumentStorageError,
    LocalDocumentStorage,
)


def test_save_document(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(tmp_path)

    stored_path = storage.save(
        "tenant-123/document-456.pdf",
        b"%PDF-1.7 test document",
    )

    assert stored_path == (
        tmp_path
        / "tenant-123"
        / "document-456.pdf"
    ).resolve()

    assert stored_path.read_bytes() == b"%PDF-1.7 test document"


def test_save_rejects_existing_storage_key(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(tmp_path)

    storage.save(
        "tenant-123/document-456.pdf",
        b"first",
    )

    with pytest.raises(
        DocumentStorageError,
        match="already exists",
    ):
        storage.save(
            "tenant-123/document-456.pdf",
            b"second",
        )


def test_storage_key_cannot_escape_root(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(tmp_path)

    with pytest.raises(
        DocumentStorageError,
        match="unsafe path segment",
    ):
        storage.save(
            "../outside.pdf",
            b"unsafe",
        )


def test_delete_document(
    tmp_path: Path,
) -> None:
    storage = LocalDocumentStorage(tmp_path)

    stored_path = storage.save(
        "tenant-123/document-456.pdf",
        b"content",
    )

    assert stored_path.exists()

    storage.delete(
        "tenant-123/document-456.pdf"
    )

    assert stored_path.exists() is False
