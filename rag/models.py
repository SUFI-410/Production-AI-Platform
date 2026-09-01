"""
SQLAlchemy ORM models for SaaS tenancy.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from rag.database import Base


class DocumentStatus(str, Enum):
    """Lifecycle status for an uploaded document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Business role of an uploaded Invoice Preflight document."""

    CONTRACT = "contract"
    SOW = "sow"
    PURCHASE_ORDER = "purchase_order"
    BILLING_INSTRUCTIONS = "billing_instructions"
    INVOICE = "invoice"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    OTHER = "other"


class PlanCode(str, Enum):
    """Subscription plan identifiers."""

    TRIAL = "trial"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"


class SubscriptionStatus(str, Enum):
    """Lifecycle state of an organization subscription."""

    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    READ_ONLY = "read_only"


class BillingInterval(str, Enum):
    """Supported paid subscription billing intervals."""

    MONTHLY = "monthly"
    ANNUAL = "annual"


class Organization(Base):
    """Tenant organization."""

    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class User(Base):
    """Application user belonging to a tenant organization."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        unique=True,
        index=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    auth_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    reset_token_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
    )

    reset_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PasswordResetRateLimit(Base):
    """Shared fixed-window limits; keys contain HMACs, not email/IP values."""

    __tablename__ = "password_reset_rate_limits"

    key_hash: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )


class Document(Base):
    """Uploaded document owned by a tenant organization."""

    __tablename__ = "documents"

    __table_args__ = (
        CheckConstraint(
            "size_bytes > 0",
            name="ck_documents_size_bytes_positive",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'processing', 'ready', 'failed')",
            name="ck_documents_status",
        ),
        CheckConstraint(
            (
                "document_type IN ("
                "'contract', "
                "'sow', "
                "'purchase_order', "
                "'billing_instructions', "
                "'invoice', "
                "'supporting_evidence', "
                "'other'"
                ")"
            ),
            name="ck_documents_document_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    uploaded_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    content_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    storage_key: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    document_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentType.OTHER.value,
        server_default=DocumentType.OTHER.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.UPLOADED.value,
        server_default=DocumentStatus.UPLOADED.value,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Plan(Base):
    """Data-driven subscription plan and entitlement limits."""

    __tablename__ = "plans"

    __table_args__ = (
        CheckConstraint(
            "invoice_checks_limit >= 0",
            name="ck_plans_invoice_checks_limit",
        ),
        CheckConstraint(
            "invoice_checks_grace >= 0",
            name="ck_plans_invoice_checks_grace",
        ),
        CheckConstraint(
            "users_limit > 0",
            name="ck_plans_users_limit",
        ),
        CheckConstraint(
            "documents_limit >= 0",
            name="ck_plans_documents_limit",
        ),
        UniqueConstraint(
            "code",
            name="uq_plans_code",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    code: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    monthly_price_cents: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    annual_price_cents: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    invoice_checks_limit: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    invoice_checks_grace: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    users_limit: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    documents_limit: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    api_access: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    audit_logs: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class Subscription(Base):
    """Organization-level subscription synchronized from Paddle."""

    __tablename__ = "subscriptions"

    __table_args__ = (
        CheckConstraint(
            (
                "status IN ("
                "'trialing', "
                "'active', "
                "'past_due', "
                "'canceled', "
                "'read_only'"
                ")"
            ),
            name="ck_subscriptions_status",
        ),
        CheckConstraint(
            (
                "billing_interval IS NULL OR "
                "billing_interval IN ('monthly', 'annual')"
            ),
            name="ck_subscriptions_billing_interval",
        ),
        UniqueConstraint(
            "organization_id",
            name="uq_subscriptions_organization_id",
        ),
        UniqueConstraint(
            "provider_subscription_id",
            name="uq_subscriptions_provider_subscription_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "plans.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    scheduled_plan_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "plans.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="paddle",
        server_default="paddle",
    )

    provider_customer_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    provider_subscription_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    billing_interval: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
    )

    current_period_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    current_period_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    cancel_at_period_end: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    trial_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    trial_ends_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    read_only_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    past_due_since: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    grace_period_ends_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class SubscriptionUsage(Base):
    """Invoice-check usage for one monthly entitlement period."""

    __tablename__ = "subscription_usage"

    __table_args__ = (
        CheckConstraint(
            "invoice_checks_used >= 0",
            name="ck_subscription_usage_invoice_checks_used",
        ),
        CheckConstraint(
            "period_end > period_start",
            name="ck_subscription_usage_period",
        ),
        UniqueConstraint(
            "subscription_id",
            "period_start",
            name="uq_subscription_usage_period",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "subscriptions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    invoice_checks_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class BillingEvent(Base):
    """Persisted billing webhook event for idempotent processing."""

    __tablename__ = "billing_events"

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_event_id",
            name="uq_billing_events_provider_event",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="paddle",
        server_default="paddle",
    )

    provider_event_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    subscription_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "subscriptions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
