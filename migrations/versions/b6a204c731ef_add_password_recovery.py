"""Add password recovery and access-token revocation state.

Revision ID: b6a204c731ef
Revises: 8e7f31c9a4d2
"""

from alembic import op
import sqlalchemy as sa

revision = "b6a204c731ef"
down_revision = "8e7f31c9a4d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column(
        "auth_version", sa.Integer(), nullable=False, server_default="0",
    ))
    op.add_column("users", sa.Column(
        "reset_token_hash", sa.String(64), nullable=True,
    ))
    op.create_unique_constraint(
        "uq_users_reset_token_hash", "users", ["reset_token_hash"],
    )
    op.add_column("users", sa.Column(
        "reset_token_expires_at", sa.DateTime(timezone=True), nullable=True,
    ))
    op.create_table(
        "password_reset_rate_limits",
        sa.Column("key_hash", sa.String(64), primary_key=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_password_reset_rate_limits_expires_at",
        "password_reset_rate_limits", ["expires_at"],
    )


def downgrade() -> None:
    op.drop_table("password_reset_rate_limits")
    op.drop_column("users", "reset_token_expires_at")
    op.drop_constraint("uq_users_reset_token_hash", "users", type_="unique")
    op.drop_column("users", "reset_token_hash")
    op.drop_column("users", "auth_version")
