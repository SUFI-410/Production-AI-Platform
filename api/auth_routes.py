"""
Authentication API routes.

Provides account registration, login, and authenticated
user profile endpoints.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.dependencies import get_current_user, get_db
from api.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from rag.models import Organization, User
from rag.security import (
    create_access_token,
    hash_password,
    verify_password,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


def _normalize_email(email: str) -> str:
    """Normalize an email address for storage and lookup."""

    return email.strip().casefold()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """
    Create an organization and its first user.

    Registration is committed atomically so an organization
    cannot be created without its initial user.
    """

    email = _normalize_email(str(request.email))

    existing_user = db.scalar(
        select(User).where(
            User.email == email
        )
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    organization = Organization(
        name=request.organization_name,
    )

    db.add(organization)

    try:
        db.flush()

        user = User(
            organization_id=organization.id,
            email=email,
            password_hash=hash_password(
                request.password
            ),
            is_active=True,
        )

        db.add(user)
        db.commit()
    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from None

    return TokenResponse(
        access_token=create_access_token(
            str(user.id)
        )
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Authenticate a user and return an access token."""

    email = _normalize_email(str(request.email))

    user = db.scalar(
        select(User).where(
            User.email == email
        )
    )

    if (
        user is None
        or not user.is_active
        or not verify_password(
            request.password,
            user.password_hash,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    return TokenResponse(
        access_token=create_access_token(
            str(user.id)
        )
    )


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> User:
    """Return the currently authenticated user."""

    return current_user
