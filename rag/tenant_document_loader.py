"""
Loading of durable tenant-owned documents for Invoice Preflight.

This layer connects PostgreSQL document metadata and durable file storage
to the existing generic DocumentLoader without exposing storage details
to the AI layer.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.document_storage import (
    DocumentStorageError,
    LocalDocumentStorage,
)
from rag.loader import DocumentLoader
from rag.models import Document as DocumentRecord


class TenantDocumentLoadError(RuntimeError):
    """Raised when a tenant document cannot be loaded safely."""


class TenantDocumentLoader:
    """
    Load durable tenant documents into LangChain Document objects.

    Generic loader metadata such as ``document_type=pdf`` is preserved.
    Invoice Preflight business metadata is stored separately under
    ``business_document_type``.
    """

    def __init__(
        self,
        storage: LocalDocumentStorage,
    ) -> None:
        self.storage = storage

    def _resolve_path(
        self,
        document: DocumentRecord,
    ):
        """Resolve and validate the durable file belonging to a record."""

        if not document.storage_key:
            raise TenantDocumentLoadError(
                "Document has no durable storage key."
            )

        try:
            path = self.storage.path_for(
                document.storage_key
            )
        except DocumentStorageError as exc:
            raise TenantDocumentLoadError(
                "Document storage key is invalid."
            ) from exc

        if not path.is_file():
            raise TenantDocumentLoadError(
                "Stored document file was not found."
            )

        return path

    @staticmethod
    def _enrich_metadata(
        documents: list[Document],
        record: DocumentRecord,
    ) -> list[Document]:
        """
        Attach tenant and Invoice Preflight metadata.

        The existing ``document_type`` value from DocumentLoader remains
        the physical source format, for example ``pdf`` or ``markdown``.
        """

        for document in documents:
            metadata = dict(
                document.metadata or {}
            )

            metadata.update(
                {
                    "tenant_document_id": str(
                        record.id
                    ),
                    "organization_id": str(
                        record.organization_id
                    ),
                    "business_document_type": (
                        record.document_type
                    ),
                    "original_filename": (
                        record.original_filename
                    ),
                }
            )

            document.metadata = metadata

        return documents

    def load(
        self,
        document: DocumentRecord,
    ) -> list[Document]:
        """Load one durable tenant document."""

        path = self._resolve_path(
            document
        )

        suffix = path.suffix.casefold()

        if suffix == ".pdf":
            loaded_documents = (
                DocumentLoader.load_pdf(
                    path
                )
            )

        elif suffix == ".md":
            loaded_documents = (
                DocumentLoader.load_markdown(
                    path
                )
            )

        else:
            raise TenantDocumentLoadError(
                "Stored document has an unsupported file type."
            )

        return self._enrich_metadata(
            loaded_documents,
            document,
        )

    def load_many(
        self,
        documents: list[DocumentRecord],
    ) -> list[Document]:
        """
        Load multiple records while preventing cross-tenant mixing.
        """

        if not documents:
            raise TenantDocumentLoadError(
                "No tenant documents were provided."
            )

        organization_ids = {
            document.organization_id
            for document in documents
        }

        if len(organization_ids) != 1:
            raise TenantDocumentLoadError(
                "Documents from multiple organizations "
                "cannot be loaded together."
            )

        loaded_documents: list[Document] = []

        for document in documents:
            loaded_documents.extend(
                self.load(
                    document
                )
            )

        return loaded_documents
