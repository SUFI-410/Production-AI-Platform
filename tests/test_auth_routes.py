from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import api.auth_routes as auth_routes_module
from api.dependencies import get_current_user, get_db
from api.main import app
from rag.models import Organization, User


ORGANIZATION_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)
USER_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)


class FakeSession:
    def __init__(
        self,
        *,
        scalar_result: Any = None,
        commit_error: Exception | None = None,
    ) -> None:
        self.scalar_result = scalar_result
        self.commit_error = commit_error

        self.added: list[Any] = []
        self.flush_called = False
        self.commit_called = False
        self.rollback_called = False

    def scalar(
        self,
        statement: Any,
    ) -> Any:
        return self.scalar_result

    def add(
        self,
        instance: Any,
    ) -> None:
        self.added.append(instance)

    def flush(self) -> None:
        self.flush_called = True

        for instance in self.added:
            if (
                isinstance(instance, Organization)
                and instance.id is None
            ):
                instance.id = ORGANIZATION_ID

    def commit(self) -> None:
        self.commit_called = True

        if self.commit_error is not None:
            raise self.commit_error

        for instance in self.added:
            if (
                isinstance(instance, User)
                and instance.id is None
            ):
                instance.id = USER_ID

    def rollback(self) -> None:
        self.rollback_called = True


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


def _override_database(
    db: FakeSession,
) -> None:
    app.dependency_overrides[get_db] = (
        lambda: cast(Session, db)
    )


def _make_user(
    *,
    is_active: bool = True,
) -> User:
    return User(
        id=USER_ID,
        organization_id=ORGANIZATION_ID,
        email="owner@example.com",
        password_hash="stored-password-hash",
        is_active=is_active,
    )


def test_register_creates_organization_and_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession()

    _override_database(db)

    monkeypatch.setattr(
        auth_routes_module,
        "hash_password",
        lambda password: "hashed-password",
    )

    monkeypatch.setattr(
        auth_routes_module,
        "create_access_token",
        lambda subject: f"token::{subject}",
    )

    client = TestClient(app)

    response = client.post(
        "/auth/register",
        json={
            "organization_name": "  Acme AI  ",
            "email": "Owner@Example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 201

    assert response.json() == {
        "access_token": f"token::{USER_ID}",
        "token_type": "bearer",
    }

    assert db.flush_called is True
    assert db.commit_called is True
    assert db.rollback_called is False

    assert len(db.added) == 2

    organization = db.added[0]
    user = db.added[1]

    assert isinstance(
        organization,
        Organization,
    )
    assert organization.id == ORGANIZATION_ID
    assert organization.name == "Acme AI"

    assert isinstance(user, User)
    assert user.id == USER_ID
    assert user.organization_id == ORGANIZATION_ID
    assert user.email == "owner@example.com"
    assert user.password_hash == "hashed-password"
    assert user.is_active is True


def test_register_rejects_duplicate_email() -> None:
    db = FakeSession(
        scalar_result=_make_user()
    )

    _override_database(db)

    client = TestClient(app)

    response = client.post(
        "/auth/register",
        json={
            "organization_name": "Acme AI",
            "email": "owner@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 409

    assert response.json() == {
        "detail": (
            "An account with this email already exists."
        )
    }

    assert db.added == []
    assert db.commit_called is False


def test_register_rolls_back_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession(
        commit_error=IntegrityError(
            "INSERT",
            {},
            Exception("duplicate"),
        )
    )

    _override_database(db)

    monkeypatch.setattr(
        auth_routes_module,
        "hash_password",
        lambda password: "hashed-password",
    )

    client = TestClient(app)

    response = client.post(
        "/auth/register",
        json={
            "organization_name": "Acme AI",
            "email": "owner@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 409
    assert db.commit_called is True
    assert db.rollback_called is True


def test_login_returns_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _make_user()
    db = FakeSession(
        scalar_result=user
    )

    _override_database(db)

    verification_calls: list[
        tuple[str, str]
    ] = []

    def fake_verify_password(
        password: str,
        password_hash: str,
    ) -> bool:
        verification_calls.append(
            (
                password,
                password_hash,
            )
        )

        return True

    monkeypatch.setattr(
        auth_routes_module,
        "verify_password",
        fake_verify_password,
    )

    monkeypatch.setattr(
        auth_routes_module,
        "create_access_token",
        lambda subject: f"token::{subject}",
    )

    client = TestClient(app)

    response = client.post(
        "/auth/login",
        json={
            "email": "Owner@Example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "access_token": f"token::{USER_ID}",
        "token_type": "bearer",
    }

    assert verification_calls == [
        (
            "StrongPassword123!",
            "stored-password-hash",
        )
    ]


def test_login_rejects_wrong_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession(
        scalar_result=_make_user()
    )

    _override_database(db)

    monkeypatch.setattr(
        auth_routes_module,
        "verify_password",
        lambda password, password_hash: False,
    )

    client = TestClient(app)

    response = client.post(
        "/auth/login",
        json={
            "email": "owner@example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid email or password."
    }

    assert response.headers[
        "www-authenticate"
    ] == "Bearer"


def test_login_rejects_inactive_user() -> None:
    db = FakeSession(
        scalar_result=_make_user(
            is_active=False
        )
    )

    _override_database(db)

    client = TestClient(app)

    response = client.post(
        "/auth/login",
        json={
            "email": "owner@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 401


def test_me_returns_authenticated_user() -> None:
    user = _make_user()

    app.dependency_overrides[
        get_current_user
    ] = lambda: user

    client = TestClient(app)

    response = client.get("/auth/me")

    assert response.status_code == 200

    assert response.json() == {
        "id": str(USER_ID),
        "organization_id": str(
            ORGANIZATION_ID
        ),
        "email": "owner@example.com",
        "is_active": True,
    }


def test_me_rejects_missing_bearer_token() -> None:
    db = FakeSession()

    _override_database(db)

    client = TestClient(app)

    response = client.get("/auth/me")

    assert response.status_code == 401

    assert response.json() == {
        "detail": (
            "Invalid or missing "
            "authentication credentials."
        )
    }
