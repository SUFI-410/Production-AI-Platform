"""
Durable storage for private tenant documents.

The initial implementation uses the local filesystem behind a small
abstraction so it can later be replaced by S3, Cloudflare R2, or another
object-storage provider without changing the document API.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from uuid import uuid4


class DocumentStorageError(RuntimeError):
    """Raised when a document cannot be stored or removed safely."""


class LocalDocumentStorage:
    """Store private uploaded documents on the local filesystem."""

    def __init__(
        self,
        root_directory: str | Path,
    ) -> None:
        self.root_directory = Path(root_directory).resolve()

    def path_for(
        self,
        storage_key: str,
    ) -> Path:
        """
        Resolve a storage key to a safe path below the storage root.

        Storage keys use POSIX-style separators regardless of the host OS.
        """

        normalized_key = storage_key.replace("\\", "/").strip("/")

        if not normalized_key:
            raise DocumentStorageError(
                "Document storage key must not be empty."
            )

        key = PurePosixPath(normalized_key)

        if any(part in {"", ".", ".."} for part in key.parts):
            raise DocumentStorageError(
                "Document storage key contains an unsafe path segment."
            )

        path = self.root_directory.joinpath(*key.parts).resolve()

        if (
            path != self.root_directory
            and self.root_directory not in path.parents
        ):
            raise DocumentStorageError(
                "Document storage key escapes the storage directory."
            )

        return path

    def save(
        self,
        storage_key: str,
        content: bytes,
    ) -> Path:
        """Persist document bytes atomically and return the stored path."""

        if not content:
            raise DocumentStorageError(
                "Cannot store an empty document."
            )

        destination = self.path_for(storage_key)

        if destination.exists():
            raise DocumentStorageError(
                "A document already exists for this storage key."
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )

        try:
            temporary_path.write_bytes(content)
            os.replace(
                temporary_path,
                destination,
            )
        except OSError as exc:
            temporary_path.unlink(
                missing_ok=True,
            )

            raise DocumentStorageError(
                "Unable to persist the document."
            ) from exc

        return destination

    def delete(
        self,
        storage_key: str,
    ) -> None:
        """Delete a stored document if it exists."""

        path = self.path_for(storage_key)

        try:
            path.unlink(
                missing_ok=True,
            )
        except OSError as exc:
            raise DocumentStorageError(
                "Unable to delete the stored document."
            ) from exc
