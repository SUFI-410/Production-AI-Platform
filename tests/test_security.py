from __future__ import annotations

import jwt
import pytest
from jwt import ExpiredSignatureError, InvalidTokenError

from rag.config import Config
from rag.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


TEST_JWT_SECRET = (
    "test-jwt-secret-key-that-is-at-least-32-bytes"
)

WRONG_JWT_SECRET = (
    "wrong-jwt-secret-key-that-is-at-least-32-bytes"
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


def test_access_token_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        TEST_JWT_SECRET,
    )

    token = create_access_token("user-123")

    payload = decode_access_token(token)

    assert payload["sub"] == "user-123"
    assert payload["type"] == "access"
    assert "iat" in payload
    assert "exp" in payload


def test_access_token_rejects_invalid_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        TEST_JWT_SECRET,
    )

    token = jwt.encode(
        {
            "sub": "user-123",
            "iat": 1,
            "exp": 4102444800,
            "type": "access",
        },
        WRONG_JWT_SECRET,
        algorithm=Config.JWT_ALGORITHM,
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(token)


def test_access_token_rejects_wrong_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        TEST_JWT_SECRET,
    )

    token = jwt.encode(
        {
            "sub": "user-123",
            "iat": 1,
            "exp": 4102444800,
            "type": "refresh",
        },
        TEST_JWT_SECRET,
        algorithm=Config.JWT_ALGORITHM,
    )

    with pytest.raises(
        InvalidTokenError,
        match="Invalid token type",
    ):
        decode_access_token(token)


def test_access_token_rejects_expired_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        TEST_JWT_SECRET,
    )

    token = jwt.encode(
        {
            "sub": "user-123",
            "iat": 1,
            "exp": 2,
            "type": "access",
        },
        TEST_JWT_SECRET,
        algorithm=Config.JWT_ALGORITHM,
    )

    with pytest.raises(ExpiredSignatureError):
        decode_access_token(token)


def test_access_token_requires_signing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="JWT_SECRET_KEY is not configured",
    ):
        create_access_token("user-123")
