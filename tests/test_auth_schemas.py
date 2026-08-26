from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from api.schemas import (
    LoginRequest,
    RegisterRequest,
    UserResponse,
)


def test_register_request_accepts_valid_input() -> None:
    request = RegisterRequest(
        invitation_token="signed-invitation",
        password="StrongPassword123!",
    )

    assert request.invitation_token == "signed-invitation"
    assert request.password == "StrongPassword123!"


def test_register_request_requires_invitation() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            password="StrongPassword123!",
        )


def test_register_request_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            invitation_token="signed-invitation",
            password="short",
        )


def test_register_request_rejects_blank_invitation() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            invitation_token="",
            password="StrongPassword123!",
        )


def test_login_request_accepts_valid_input() -> None:
    request = LoginRequest(
        email="owner@example.com",
        password="StrongPassword123!",
    )

    assert request.email == "owner@example.com"
    assert request.password == "StrongPassword123!"


def test_user_response_reads_attributes() -> None:
    class FakeUser:
        id = UUID(
            "11111111-1111-1111-1111-111111111111"
        )
        organization_id = UUID(
            "22222222-2222-2222-2222-222222222222"
        )
        email = "owner@example.com"
        is_active = True

    response = UserResponse.model_validate(
        FakeUser()
    )

    assert response.id == FakeUser.id
    assert response.organization_id == (
        FakeUser.organization_id
    )
    assert response.email == "owner@example.com"
    assert response.is_active is True
