from __future__ import annotations

import jwt
import pytest
from jwt import InvalidTokenError

from rag.config import Config
from rag.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_uses_argon2() -> None:
    password_hash = hash_password("TestPassword123!")

    assert password_hash.startswith("$argon2")


def test_verify_password_accepts_correct_password() -> None:
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert verify_password(
        password,
        password_hash,
    )


def test_verify_password_rejects_wrong_password() -> None:
    password_hash = hash_password("TestPassword123!")

    assert not verify_password(
        "WrongPassword!",
        password_hash,
    )


def test_access_token_round_trip() -> None:
    token = create_access_token("user-123")

    payload = decode_access_token(token)

    assert payload["sub"] == "user-123"
    assert payload["type"] == "access"
    assert "iat" in payload
    assert "exp" in payload


def test_access_token_rejects_invalid_signature() -> None:
    token = jwt.encode(
        {
            "sub": "user-123",
            "iat": 1,
            "exp": 4102444800,
            "type": "access",
        },
        "x" * 32,
        algorithm=Config.JWT_ALGORITHM,
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(token)


def test_access_token_rejects_wrong_type() -> None:
    if not Config.JWT_SECRET_KEY:
        pytest.fail("JWT_SECRET_KEY is not configured.")

    token = jwt.encode(
        {
            "sub": "user-123",
            "iat": 1,
            "exp": 4102444800,
            "type": "refresh",
        },
        Config.JWT_SECRET_KEY,
        algorithm=Config.JWT_ALGORITHM,
    )

    with pytest.raises(
        InvalidTokenError,
        match="Invalid token type",
    ):
        decode_access_token(token)
