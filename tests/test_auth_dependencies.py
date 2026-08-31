from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt import InvalidTokenError
from sqlalchemy.orm import Session

import api.dependencies as dependencies_module
from api.dependencies import (
    get_current_organization,
    get_current_user,
)
from rag.models import Organization, User


USER_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)

ORGANIZATION_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)


class FakeUser:
    def __init__(
        self,
        *,
        is_active: bool = True,
        organization_id: UUID = ORGANIZATION_ID,
    ) -> None:
        self.id = USER_ID
        self.organization_id = organization_id
        self.is_active = is_active
        self.auth_version = 0


class FakeSession:
    def __init__(
        self,
        user: FakeUser | None,
    ) -> None:
        self.user = user
        self.requested_user_id: UUID | None = None
        self.requested_model: Any = None

    def get(
        self,
        model: Any,
        user_id: UUID,
    ) -> FakeUser | None:
        self.requested_model = model
        self.requested_user_id = user_id

        return self.user


def _credentials(
    token: str = "test-token",
) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=token,
    )


def test_get_current_user_accepts_valid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = FakeUser()
    db = FakeSession(user)

    monkeypatch.setattr(
        dependencies_module,
        "decode_access_token",
        lambda token: {
            "sub": str(USER_ID),
            "type": "access",
        },
    )

    result = get_current_user(
        credentials=_credentials(),
        db=cast(Session, db),
    )

    assert result is user
    assert db.requested_user_id == USER_ID


def test_get_current_user_rejects_missing_credentials() -> None:
    db = FakeSession(None)

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(
            credentials=None,
            db=cast(Session, db),
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {
        "WWW-Authenticate": "Bearer"
    }


@pytest.mark.parametrize("version", [True, "0", None, -1, 1])
def test_get_current_user_rejects_wrong_token_version(monkeypatch, version):
    monkeypatch.setattr(dependencies_module, "decode_access_token", lambda token: {
        "sub": str(USER_ID), "type": "access", "auth_version": version,
    })
    with pytest.raises(HTTPException) as error:
        get_current_user(_credentials(), cast(Session, FakeSession(FakeUser())))
    assert error.value.status_code == 401


def test_legacy_access_token_is_rejected_after_reset(monkeypatch):
    user = FakeUser()
    user.auth_version = 1
    monkeypatch.setattr(dependencies_module, "decode_access_token", lambda token: {
        "sub": str(USER_ID), "type": "access",
    })
    with pytest.raises(HTTPException):
        get_current_user(_credentials(), cast(Session, FakeSession(user)))


def test_get_current_user_rejects_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession(None)

    def reject_token(
        token: str,
    ) -> dict[str, object]:
        raise InvalidTokenError("invalid token")

    monkeypatch.setattr(
        dependencies_module,
        "decode_access_token",
        reject_token,
    )

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(
            credentials=_credentials(),
            db=cast(Session, db),
        )

    assert exc_info.value.status_code == 401


def test_get_current_user_rejects_invalid_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession(None)

    monkeypatch.setattr(
        dependencies_module,
        "decode_access_token",
        lambda token: {
            "sub": "not-a-uuid",
            "type": "access",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(
            credentials=_credentials(),
            db=cast(Session, db),
        )

    assert exc_info.value.status_code == 401


def test_get_current_user_rejects_unknown_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession(None)

    monkeypatch.setattr(
        dependencies_module,
        "decode_access_token",
        lambda token: {
            "sub": str(USER_ID),
            "type": "access",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(
            credentials=_credentials(),
            db=cast(Session, db),
        )

    assert exc_info.value.status_code == 401


def test_get_current_user_rejects_inactive_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession(
        FakeUser(is_active=False)
    )

    monkeypatch.setattr(
        dependencies_module,
        "decode_access_token",
        lambda token: {
            "sub": str(USER_ID),
            "type": "access",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(
            credentials=_credentials(),
            db=cast(Session, db),
        )

    assert exc_info.value.status_code == 401


def test_get_current_organization_returns_user_organization() -> None:
    user = FakeUser()

    organization = Organization(
        id=ORGANIZATION_ID,
        name="Acme AI",
    )

    class FakeOrganizationSession:
        def get(
            self,
            model: Any,
            object_id: UUID,
        ) -> Organization | None:
            assert model is Organization
            assert object_id == ORGANIZATION_ID

            return organization

    db = FakeOrganizationSession()

    result = get_current_organization(
        current_user=cast(User, user),
        db=cast(Session, db),
    )

    assert result is organization


def test_get_current_organization_rejects_missing_organization() -> None:
    user = FakeUser()

    class FakeOrganizationSession:
        def get(
            self,
            model: Any,
            object_id: UUID,
        ) -> Organization | None:
            assert model is Organization
            assert object_id == ORGANIZATION_ID

            return None

    db = FakeOrganizationSession()

    with pytest.raises(HTTPException) as exc_info:
        get_current_organization(
            current_user=cast(User, user),
            db=cast(Session, db),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == (
        "Authenticated user is not assigned "
        "to a valid organization."
    )
