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


def create_access_token(subject: str) -> str:
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
    }

    return jwt.encode(
        payload,
        _get_jwt_secret_key(),
        algorithm=Config.JWT_ALGORITHM,
    )


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
