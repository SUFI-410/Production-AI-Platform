"""add document storage fields

Revision ID: 8e7f31c9a4d2
Revises: cdb195b848b9
Create Date: 2026-08-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8e7f31c9a4d2"
down_revision: Union[str, Sequence[str], None] = "cdb195b848b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add durable storage metadata and Invoice Preflight document type."""

    op.add_column(
        "documents",
        sa.Column(
            "storage_key",
            sa.String(length=512),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "document_type",
            sa.String(length=32),
            server_default="other",
            nullable=False,
        ),
    )

    op.create_check_constraint(
        "ck_documents_document_type",
        "documents",
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
    )


def downgrade() -> None:
    """Remove document storage metadata and document type."""

    op.drop_constraint(
        "ck_documents_document_type",
        "documents",
        type_="check",
    )

    op.drop_column(
        "documents",
        "document_type",
    )

    op.drop_column(
        "documents",
        "storage_key",
    )
