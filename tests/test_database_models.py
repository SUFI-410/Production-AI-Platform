from __future__ import annotations

from rag.database import Base
from rag.models import Organization, User


def test_tenancy_tables_are_registered() -> None:
    assert "organizations" in Base.metadata.tables
    assert "users" in Base.metadata.tables


def test_user_references_organization() -> None:
    foreign_keys = list(User.__table__.foreign_keys)

    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "organizations.id"


def test_expected_table_names() -> None:
    assert Organization.__tablename__ == "organizations"
    assert User.__tablename__ == "users"
