"""
Authentication security utilities.
"""

from __future__ import annotations

from pwdlib import PasswordHash


_password_hasher = PasswordHash.recommended()


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
