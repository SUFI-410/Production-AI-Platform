"""
Authentication security utilities.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from rag.config import Config


_password_hasher = PasswordHash.recommended()


def _get_jwt_secret_key() -> str:
    """Return the configured JWT signing secret."""

    if not Config.JWT_SECRET_KEY:
        raise RuntimeError(
            "JWT_SECRET_KEY is not configured."
        )

    return Config.JWT_SECRET_KEY


def hash_password(password: str) -> str:
    """Hash a plaintext password for secure storage."""

    return _password_hasher.hash(password)


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    """Verify a plaintext password against its stored hash."""

    return _password_hasher.verify(
        password,
        password_hash,
    )


def create_access_token(subject: str, auth_version: int = 0) -> str:
    """Create a signed JWT access token."""

    now = datetime.now(timezone.utc)

    expires_at = now + timedelta(
        minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": subject,
        "iat": now,
        "exp": expires_at,
        "type": "access",
        "auth_version": auth_version,
    }

    return jwt.encode(
        payload,
        _get_jwt_secret_key(),
        algorithm=Config.JWT_ALGORITHM,
    )


def create_registration_invite(
    email: str,
    organization_name: str,
    *,
    expires_hours: int = 72,
) -> str:
    """Create a signed, expiring pilot-registration invitation."""

    normalized_email = email.strip().casefold()
    normalized_organization = organization_name.strip()

    if not normalized_email:
        raise ValueError("Invitation email is required.")

    if len(normalized_organization) < 2:
        raise ValueError(
            "Invitation organization name must contain at least 2 characters."
        )

    if expires_hours < 1:
        raise ValueError(
            "Invitation expiry must be at least 1 hour."
        )

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=expires_hours)

    payload = {
        "sub": normalized_email,
        "organization_name": normalized_organization,
        "iat": now,
        "exp": expires_at,
        "type": "registration_invite",
    }

    return jwt.encode(
        payload,
        _get_jwt_secret_key(),
        algorithm=Config.JWT_ALGORITHM,
    )


def decode_registration_invite(
    token: str,
) -> dict[str, object]:
    """Decode and validate a pilot-registration invitation."""

    payload = jwt.decode(
        token,
        _get_jwt_secret_key(),
        algorithms=[Config.JWT_ALGORITHM],
        options={
            "require": [
                "sub",
                "iat",
                "exp",
                "type",
            ]
        },
    )

    if payload.get("type") != "registration_invite":
        raise InvalidTokenError(
            "Invalid token type."
        )

    subject = payload.get("sub")
    organization_name = payload.get(
        "organization_name"
    )

    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError(
            "Invalid invitation email."
        )

    if (
        not isinstance(organization_name, str)
        or len(organization_name.strip()) < 2
    ):
        raise InvalidTokenError(
            "Invalid invitation organization."
        )

    return payload


def decode_access_token(
    token: str,
) -> dict[str, object]:
    """Decode and validate a JWT access token."""

    payload = jwt.decode(
        token,
        _get_jwt_secret_key(),
        algorithms=[Config.JWT_ALGORITHM],
        options={
            "require": [
                "sub",
                "iat",
                "exp",
                "type",
            ]
        },
    )

    if payload.get("type") != "access":
        raise InvalidTokenError(
            "Invalid token type."
        )

    subject = payload.get("sub")

    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError(
            "Invalid token subject."
        )

    return payload
