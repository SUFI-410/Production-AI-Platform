"""Add provider updated at to subscriptions.

Revision ID: 92b350c8ac0c
Revises: 27ee514ca46a
"""

from alembic import op
import sqlalchemy as sa


revision = "92b350c8ac0c"
down_revision = "27ee514ca46a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column(
            "provider_updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "subscriptions",
        "provider_updated_at",
    )
