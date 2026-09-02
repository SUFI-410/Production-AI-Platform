"""Seed subscription plans.

Revision ID: 27ee514ca46a
Revises: 0d5d642cec16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "27ee514ca46a"
down_revision = "0d5d642cec16"
branch_labels = None
depends_on = None


TRIAL_PLAN_ID = "11111111-1111-4111-8111-111111111111"
STARTER_PLAN_ID = "22222222-2222-4222-8222-222222222222"
PROFESSIONAL_PLAN_ID = "33333333-3333-4333-8333-333333333333"
BUSINESS_PLAN_ID = "44444444-4444-4444-8444-444444444444"


def upgrade() -> None:
    plans = sa.table(
        "plans",
        sa.column(
            "id",
            postgresql.UUID(as_uuid=True),
        ),
        sa.column(
            "code",
            sa.String(),
        ),
        sa.column(
            "name",
            sa.String(),
        ),
        sa.column(
            "monthly_price_cents",
            sa.Integer(),
        ),
        sa.column(
            "annual_price_cents",
            sa.Integer(),
        ),
        sa.column(
            "invoice_checks_limit",
            sa.Integer(),
        ),
        sa.column(
            "invoice_checks_grace",
            sa.Integer(),
        ),
        sa.column(
            "users_limit",
            sa.Integer(),
        ),
        sa.column(
            "documents_limit",
            sa.Integer(),
        ),
        sa.column(
            "api_access",
            sa.Boolean(),
        ),
        sa.column(
            "audit_logs",
            sa.Boolean(),
        ),
        sa.column(
            "is_active",
            sa.Boolean(),
        ),
    )

    op.bulk_insert(
        plans,
        [
            {
                "id": TRIAL_PLAN_ID,
                "code": "trial",
                "name": "Trial",
                "monthly_price_cents": None,
                "annual_price_cents": None,
                "invoice_checks_limit": 25,
                "invoice_checks_grace": 0,
                "users_limit": 1,
                "documents_limit": 3,
                "api_access": False,
                "audit_logs": False,
                "is_active": True,
            },
            {
                "id": STARTER_PLAN_ID,
                "code": "starter",
                "name": "Starter",
                "monthly_price_cents": 14900,
                "annual_price_cents": 149000,
                "invoice_checks_limit": 250,
                "invoice_checks_grace": 25,
                "users_limit": 3,
                "documents_limit": 50,
                "api_access": False,
                "audit_logs": False,
                "is_active": True,
            },
            {
                "id": PROFESSIONAL_PLAN_ID,
                "code": "professional",
                "name": "Professional",
                "monthly_price_cents": 39900,
                "annual_price_cents": 399000,
                "invoice_checks_limit": 1000,
                "invoice_checks_grace": 100,
                "users_limit": 10,
                "documents_limit": 500,
                "api_access": True,
                "audit_logs": True,
                "is_active": True,
            },
            {
                "id": BUSINESS_PLAN_ID,
                "code": "business",
                "name": "Business",
                "monthly_price_cents": 79900,
                "annual_price_cents": 799000,
                "invoice_checks_limit": 5000,
                "invoice_checks_grace": 500,
                "users_limit": 25,
                "documents_limit": 2500,
                "api_access": True,
                "audit_logs": True,
                "is_active": True,
            },
        ],
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM plans
            WHERE code IN (
                'trial',
                'starter',
                'professional',
                'business'
            )
            """
        )
    )
