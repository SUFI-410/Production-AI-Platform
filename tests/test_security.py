from __future__ import annotations

from rag.security import hash_password, verify_password


def test_hash_password_uses_argon2() -> None:
    password_hash = hash_password("TestPassword123!")

    assert password_hash.startswith("$argon2")


def test_verify_password_accepts_correct_password() -> None:
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert verify_password(password, password_hash)


def test_verify_password_rejects_wrong_password() -> None:
    password_hash = hash_password("TestPassword123!")

    assert not verify_password(
        "WrongPassword!",
        password_hash,
    )
