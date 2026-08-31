"""Public account recovery endpoints; no login required."""

from typing import Annotated

from fastapi import (
    APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response,
)
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.turnstile import verify_turnstile
from rag.password_reset import (
    consume_limit,
    deliver_reset_confirmation,
    deliver_reset_email,
    reset_password,
    validate_reset_configuration,
)

router = APIRouter(prefix="/auth/password-reset", tags=["Authentication"])
GENERIC_MESSAGE = (
    "If an active account matches that email, a reset link will be sent. "
    "Check your inbox and spam folder. If you requested several links, "
    "wait an hour before trying again."
)


class ResetEmailRequest(BaseModel):
    email: EmailStr
    turnstile_token: str = Field(min_length=1, max_length=2048)


class ResetPasswordRequest(BaseModel):
    token: str = Field(
        min_length=43, max_length=43, pattern=r"^[A-Za-z0-9_-]{43}$",
    )
    password: str = Field(min_length=12, max_length=128)


class ResetResponse(BaseModel):
    message: str


def require_reset_configuration() -> None:
    try:
        validate_reset_configuration()
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Password recovery is temporarily unavailable. Please contact support."
            ),
        ) from None


def check_ip_limit(db: Session, request: Request, scope: str, limit: int) -> None:
    # Only use ASGI's client address. Uvicorn must trust only the actual proxy,
    # never arbitrary forwarded headers from public clients.
    address = request.client.host if request.client else "unknown"
    if not consume_limit(db, scope, address, limit, 900):
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait 15 minutes and try again.",
            headers={"Retry-After": "900"},
        )


@router.post("/request", response_model=ResetResponse, status_code=202)
def request_reset(
    payload: ResetEmailRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
) -> ResetResponse:
    require_reset_configuration()
    check_ip_limit(db, request, "request-ip", 10)
    verify_turnstile(payload.turnstile_token, request)
    email = str(payload.email).strip().casefold()
    # Same limiter and response for existing, unknown and inactive accounts.
    if consume_limit(db, "request-email", email, 3, 3600):
        background_tasks.add_task(deliver_reset_email, email)
    response.headers["Cache-Control"] = "no-store"
    return ResetResponse(message=GENERIC_MESSAGE)


@router.post("/confirm", response_model=ResetResponse)
def confirm_reset(
    payload: ResetPasswordRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
) -> ResetResponse:
    require_reset_configuration()
    check_ip_limit(db, request, "confirm-ip", 20)
    email = reset_password(db, payload.token, payload.password)
    if email is None:
        raise HTTPException(
            status_code=400,
            detail="This reset link is invalid or expired. Please request a new link.",
        )
    background_tasks.add_task(deliver_reset_confirmation, email)
    response.headers["Cache-Control"] = "no-store"
    return ResetResponse(
        message=(
            "Password updated. Sign in with your new password. "
            "Previous sessions were revoked."
        ),
    )
