"""
Tenant document upload API routes.

Phase 2.3 validates uploaded files and creates tenant-owned
document metadata. Durable raw-file storage is added in Phase 2.4.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import (
    get_current_organization,
    get_current_user,
    get_db,
)
from api.schemas import DocumentResponse
from rag.models import (
    Document as DocumentRecord,
    DocumentStatus,
    Organization,
    User,
)


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


def _safe_filename(raw_filename: str | None) -> str:
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
            detail="The uploaded file has an unsupported content type.",
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
            detail="The uploaded file exceeds the 10 MiB limit.",
        )

    if suffix == ".pdf":
        if b"%PDF-" not in content[:1024]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is not a valid PDF.",
            )

    if suffix == ".md":
        try:
            markdown = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Markdown files must use UTF-8 encoding.",
            ) from None

        if not markdown.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded Markdown file is empty.",
            )

    return filename, content


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
) -> DocumentRecord:
    """
    Validate a private document and create its tenant-owned record.

    Organization ownership is derived from authenticated server-side
    state. Clients cannot choose the owning organization.
    """

    filename, content = await _validate_upload(file)

    document = DocumentRecord(
        organization_id=current_organization.id,
        uploaded_by_user_id=current_user.id,
        original_filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        status=DocumentStatus.UPLOADED.value,
    )

    db.add(document)

    try:
        db.commit()
        db.refresh(document)
    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create the document record.",
        ) from None

    return document
