"""
Tenant document upload API routes.

Uploaded tenant documents are validated, stored durably, and recorded
in PostgreSQL for later Invoice Preflight processing.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import (
    get_current_organization,
    get_current_user,
    get_db,
)
from api.schemas import DocumentResponse
from rag.billing_service import (
    BillingService,
    BillingServiceError,
)
from rag.config import Config
from rag.document_storage import (
    DocumentStorageError,
    LocalDocumentStorage,
)
from rag.models import (
    Document as DocumentRecord,
    DocumentStatus,
    DocumentType,
    Organization,
    User,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


MAX_DOCUMENT_UPLOAD_BYTES = 10 * 1024 * 1024

_ALLOWED_CONTENT_TYPES: dict[str, set[str]] = {
    ".pdf": {
        "application/pdf",
        "application/octet-stream",
    },
    ".md": {
        "text/markdown",
        "text/plain",
        "application/octet-stream",
    },
}


def get_document_storage() -> LocalDocumentStorage:
    """Return the configured durable document storage backend."""

    return LocalDocumentStorage(
        Config.DOCUMENT_STORAGE_DIR,
    )


def get_billing_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> BillingService:
    """Return document-upload billing enforcement."""

    return BillingService(db)


def _safe_filename(
    raw_filename: str | None,
) -> str:
    """Return a safe basename for an uploaded filename."""

    if raw_filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    normalized = raw_filename.replace("\\", "/")
    filename = PurePosixPath(normalized).name.strip()

    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid filename is required.",
        )

    if len(filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must not exceed 255 characters.",
        )

    return filename


async def _validate_upload(
    file: UploadFile,
) -> tuple[str, bytes]:
    """Read and validate an uploaded PDF or Markdown file."""

    filename = _safe_filename(file.filename)
    suffix = PurePosixPath(filename).suffix.casefold()

    allowed_content_types = _ALLOWED_CONTENT_TYPES.get(suffix)

    if allowed_content_types is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and Markdown files are supported.",
        )

    content_type = (file.content_type or "").casefold()

    if content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=("The uploaded file has an unsupported content type."),
        )

    content = await file.read(MAX_DOCUMENT_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_DOCUMENT_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=("The uploaded file exceeds the 10 MiB limit."),
        )

    if suffix == ".pdf":
        if b"%PDF-" not in content[:1024]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=("The uploaded file is not a valid PDF."),
            )

    if suffix == ".md":
        try:
            markdown = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=("Markdown files must use UTF-8 encoding."),
            ) from None

        if not markdown.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=("The uploaded Markdown file is empty."),
            )

    return filename, content


@router.get(
    "",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK,
)
def list_documents(
    current_organization: Annotated[
        Organization,
        Depends(get_current_organization),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> list[DocumentRecord]:
    """List documents owned by the authenticated organization."""

    statement = (
        select(DocumentRecord)
        .where(DocumentRecord.organization_id == current_organization.id)
        .order_by(DocumentRecord.created_at.desc())
    )

    try:
        return list(db.scalars(statement).all())
    except SQLAlchemyError:
        logger.exception("Unable to list tenant documents.")

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail="Unable to list documents.",
        ) from None


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: UUID,
    current_organization: Annotated[
        Organization,
        Depends(get_current_organization),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    document_storage: Annotated[
        LocalDocumentStorage,
        Depends(get_document_storage),
    ],
) -> Response:
    """Delete one document owned by the authenticated organization."""

    statement = select(DocumentRecord).where(
        DocumentRecord.id == document_id,
        DocumentRecord.organization_id == current_organization.id,
    )

    try:
        document = db.scalar(statement)
    except SQLAlchemyError:
        logger.exception("Unable to resolve tenant document for deletion.")

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail="Unable to delete the document.",
        ) from None

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.storage_key:
        try:
            document_storage.delete(document.storage_key)
        except DocumentStorageError:
            logger.exception("Unable to remove stored tenant document.")

            raise HTTPException(
                status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
                detail="Unable to delete the document.",
            ) from None

    db.delete(document)

    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()

        logger.exception("Unable to delete tenant document record.")

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail="Unable to delete the document.",
        ) from None

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(
            description=(
                "PDF or Markdown document belonging to the authenticated organization."
            ),
        ),
    ],
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
    current_organization: Annotated[
        Organization,
        Depends(get_current_organization),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    billing_service: Annotated[
        BillingService,
        Depends(get_billing_service),
    ],
    document_storage: Annotated[
        LocalDocumentStorage,
        Depends(get_document_storage),
    ],
    document_type: Annotated[
        DocumentType,
        Form(
            description=("Business role of the uploaded document."),
        ),
    ] = DocumentType.OTHER,
) -> DocumentRecord:
    """
    Validate and durably store a private tenant document.

    Organization ownership is derived from authenticated server-side
    state. Clients cannot choose the owning organization.
    """

    filename, content = await _validate_upload(file)

    try:
        billing_service.ensure_document_upload_allowed(
            current_organization.id
        )
    except BillingServiceError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Document upload is not allowed by "
                "the current billing entitlement."
            ),
        ) from None

    document_id = uuid4()

    suffix = PurePosixPath(filename).suffix.casefold()

    storage_key = f"{current_organization.id}/{document_id}{suffix}"

    try:
        document_storage.save(
            storage_key,
            content,
        )
    except DocumentStorageError:
        db.rollback()

        logger.exception("Unable to persist uploaded document.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to store the document.",
        ) from None

    document = DocumentRecord(
        id=document_id,
        organization_id=current_organization.id,
        uploaded_by_user_id=current_user.id,
        original_filename=filename,
        content_type=(file.content_type or "application/octet-stream"),
        size_bytes=len(content),
        storage_key=storage_key,
        document_type=document_type.value,
        status=DocumentStatus.UPLOADED.value,
    )

    db.add(document)

    try:
        db.commit()
        db.refresh(document)
    except SQLAlchemyError:
        db.rollback()

        try:
            document_storage.delete(storage_key)
        except DocumentStorageError:
            logger.exception(
                "Unable to remove orphaned uploaded document after database failure."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=("Unable to create the document record."),
        ) from None

    return document
