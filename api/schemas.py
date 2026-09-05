"""
Pydantic request and response schemas for the Production AI Platform API.

Responsibilities:

- Validate incoming client requests.
- Define consistent API response models.
- Provide automatic OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)
from rag.models import DocumentType


class RegisterRequest(BaseModel):
    """
    Request payload for activating an invited pilot workspace.
    """

    invitation_token: str = Field(
        ...,
        min_length=1,
        description=(
            "Signed pilot invitation supplied by the workspace owner."
        ),
    )

    password: str = Field(
        ...,
        min_length=12,
        max_length=128,
        description=(
            "Password for the new account. Must contain at least 12 characters."
        ),
    )

class LoginRequest(BaseModel):
    """
    Request payload for authenticating an existing user.
    """

    email: EmailStr = Field(
        ...,
        description="Account email address.",
        examples=["owner@example.com"],
    )

    password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Account password.",
    )


class TokenResponse(BaseModel):
    """
    Bearer access token returned after successful authentication.
    """

    access_token: str = Field(
        ...,
        description="Signed JWT access token.",
    )

    token_type: str = Field(
        default="bearer",
        description="Authentication scheme for the token.",
    )


class UserResponse(BaseModel):
    """
    Public representation of an authenticated user.
    """

    id: UUID
    organization_id: UUID
    email: EmailStr
    is_active: bool

    model_config = ConfigDict(
        from_attributes=True,
    )


class DocumentResponse(BaseModel):
    """
    Public metadata for a tenant-owned uploaded document.

    Internal storage locations are intentionally not exposed.
    """

    id: UUID
    organization_id: UUID
    uploaded_by_user_id: UUID | None
    original_filename: str
    content_type: str
    size_bytes: int
    document_type: DocumentType
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


class BillingRequirementsExtractRequest(BaseModel):
    """
    Request for extracting billing requirements from tenant documents.
    """

    document_ids: list[UUID] = Field(
        ...,
        min_length=1,
        max_length=20,
        description=(
            "Tenant-owned Contract, SOW, Purchase Order, or "
            "Billing Instructions document IDs to analyze."
        ),
    )

    @field_validator("document_ids")
    @classmethod
    def validate_unique_document_ids(
        cls,
        value: list[UUID],
    ) -> list[UUID]:
        """Reject duplicate document IDs."""

        if len(value) != len(set(value)):
            raise ValueError(
                "Document IDs must be unique."
            )

        return value

    model_config = ConfigDict(
        extra="forbid",
    )


class InvoicePreflightRequest(BaseModel):
    """
    Request for evaluating one invoice against billing documents.
    """

    billing_document_ids: list[UUID] = Field(
        ...,
        min_length=1,
        max_length=20,
        description=(
            "Tenant-owned Contract, SOW, Purchase Order, or "
            "Billing Instructions document IDs."
        ),
    )

    invoice_document_id: UUID = Field(
        ...,
        description=(
            "Tenant-owned invoice document ID to evaluate."
        ),
    )

    @field_validator("billing_document_ids")
    @classmethod
    def validate_unique_billing_document_ids(
        cls,
        value: list[UUID],
    ) -> list[UUID]:
        """Reject duplicate billing-document IDs."""

        if len(value) != len(set(value)):
            raise ValueError(
                "Billing document IDs must be unique."
            )

        return value

    @model_validator(mode="after")
    def validate_invoice_is_separate(
        self,
    ) -> InvoicePreflightRequest:
        """Prevent one document from serving both input roles."""

        if (
            self.invoice_document_id
            in self.billing_document_ids
        ):
            raise ValueError(
                "Invoice document ID must not appear in "
                "billing document IDs."
            )

        return self

    model_config = ConfigDict(
        extra="forbid",
    )


class ChatRequest(BaseModel):
    """
    Request payload for the chat endpoint.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question.",
        examples=["Explain Python decorators."],
    )

    turnstile_token: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description=(
            "Single-use Cloudflare Turnstile verification token."
        ),
    )

    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Optional conversation identifier. "
            "A server-generated identifier is returned "
            "when omitted."
        ),
        examples=["user-123"],
    )

    use_cache: bool = Field(
        default=True,
        description="Whether cached responses may be used.",
    )


class Source(BaseModel):
    """
    Information about a retrieved source document.
    """

    document: str = Field(
        ...,
        description="Document filename or identifier.",
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cross-encoder relevance score.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document metadata.",
    )


class ChatResponse(BaseModel):
    """
    Response returned from the chat endpoint.
    """

    answer: str = Field(
        ...,
        description="Final answer generated by the LLM.",
    )

    sources: list[Source] = Field(
        default_factory=list,
        description="Retrieved supporting sources.",
    )

    session_id: str = Field(
        ...,
        description=(
            "Conversation identifier used for this request. "
            "Send it with subsequent requests to preserve "
            "history."
        ),
    )

    cached: bool = Field(
        default=False,
        description="True if the answer came from the response cache.",
    )

    grounded: bool = Field(
        default=False,
        description=(
            "Whether the answer is supported by returned "
            "knowledge-base sources and is not a refusal."
        ),
    )

    latency_ms: float = Field(
        ...,
        ge=0,
        description="End-to-end processing time in milliseconds.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "answer": (
                    "A decorator is a callable that wraps another function."
                ),
                "sources": [
                    {
                        "document": "python_decorators.md",
                        "score": 0.97,
                        "metadata": {
                            "page": 1,
                            "source": "docs",
                        },
                    }
                ],
                "session_id": "user-123",
                "cached": False,
                "grounded": True,
                "latency_ms": 842.6,
            }
        },
    )


class PaddleCheckoutRequest(BaseModel):
    """
    Request for creating a Paddle subscription checkout.
    """

    plan_code: str = Field(
        ...,
        description="Paid subscription plan.",
        examples=["starter"],
    )

    billing_interval: str = Field(
        ...,
        description="Subscription billing interval.",
        examples=["monthly"],
    )

    @field_validator("plan_code")
    @classmethod
    def validate_plan_code(
        cls,
        value: str,
    ) -> str:
        """Allow only paid self-service plans."""

        allowed = {
            "starter",
            "professional",
            "business",
        }

        if value not in allowed:
            raise ValueError(
                "Plan must be starter, professional, or business."
            )

        return value

    @field_validator("billing_interval")
    @classmethod
    def validate_billing_interval(
        cls,
        value: str,
    ) -> str:
        """Allow supported subscription billing intervals."""

        allowed = {
            "monthly",
            "annual",
        }

        if value not in allowed:
            raise ValueError(
                "Billing interval must be monthly or annual."
            )

        return value

    model_config = ConfigDict(
        extra="forbid",
    )


class PaddleCheckoutResponse(BaseModel):
    """
    Paddle checkout transaction returned to the frontend.
    """

    transaction_id: str = Field(
        ...,
        min_length=1,
        description="Paddle transaction identifier.",
    )

    checkout_url: str = Field(
        ...,
        min_length=1,
        description="Paddle checkout URL for the transaction.",
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class BillingStatusResponse(BaseModel):
    """Current subscription, usage, and plan capacities."""

    organization_id: UUID
    subscription_id: UUID
    plan_code: str
    plan_name: str
    subscription_status: str
    access_mode: str
    billing_interval: str | None
    current_period_start: datetime | None
    current_period_end: datetime | None
    cancel_at_period_end: bool
    invoice_checks_used: int
    invoice_checks_limit: int
    invoice_checks_grace: int
    can_run_invoice_check: bool
    usage_period_start: datetime
    usage_period_end: datetime
    documents_used: int
    documents_limit: int
    can_upload_document: bool
    users_used: int
    users_limit: int
    api_access: bool
    audit_logs: bool

    model_config = ConfigDict(
        extra="forbid",
    )


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = Field(
        ...,
        examples=["healthy"],
    )

    version: str = Field(
        ...,
        examples=["1.0.0"],
    )


class ErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    detail: str = Field(
        ...,
        description="Human-readable error message.",
    )


class PaddlePortalResponse(BaseModel):
    """Temporary authenticated Paddle portal link."""

    portal_url: str = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")
