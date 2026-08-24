"""
Dependency injection for the FastAPI application.

This module manages application-level and request-level
dependencies used by API routes.
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import Lock
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from sqlalchemy.orm import Session

from rag.application import RAGApplication
from rag.database import SessionLocal
from rag.logger import get_logger
from rag.models import Organization, User
from rag.security import decode_access_token


logger = get_logger(__name__)

_rag_application: RAGApplication | None = None
_initialization_lock = Lock()

_bearer_scheme = HTTPBearer(auto_error=False)


def get_db() -> Iterator[Session]:
    """
    Provide a database session for one request.

    The session is always closed after the request finishes,
    including when an exception occurs.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def _unauthorized() -> HTTPException:
    """Return the standard authentication failure response."""

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_scheme),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """
    Resolve the authenticated user from a Bearer access token.

    The JWT subject contains only the user ID. Organization and
    authorization state are resolved from PostgreSQL.
    """

    if credentials is None:
        raise _unauthorized()

    try:
        payload = decode_access_token(
            credentials.credentials
        )
        subject = payload["sub"]
        user_id = UUID(str(subject))
    except (
        InvalidTokenError,
        KeyError,
        TypeError,
        ValueError,
    ):
        raise _unauthorized() from None

    user = db.get(User, user_id)

    if user is None or not user.is_active:
        raise _unauthorized()

    return user


def get_current_organization(
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> Organization:
    """
    Resolve the authenticated user's organization.

    Organization membership comes from the authenticated
    PostgreSQL user record rather than from JWT claims.
    """

    organization = db.get(
        Organization,
        current_user.organization_id,
    )

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Authenticated user is not assigned "
                "to a valid organization."
            ),
        )

    return organization


def get_rag_application() -> RAGApplication:
    """
    Return the singleton RAG application.

    Initialization is protected by a process-level lock so that
    concurrent requests cannot initialize the RAG application
    more than once.
    """

    global _rag_application

    if _rag_application is not None:
        return _rag_application

    with _initialization_lock:
        if _rag_application is not None:
            return _rag_application

        logger.info("Initializing RAG application...")

        application = RAGApplication()

        try:
            application.load_existing()
        except Exception:
            logger.exception(
                "Failed to initialize the RAG application."
            )
            raise

        _rag_application = application

        logger.info(
            "RAG application initialized successfully."
        )

        return _rag_application


def get_application() -> RAGApplication:
    """
    FastAPI dependency.

    Return the shared, fully initialized RAG application.
    """

    return get_rag_application()
