"""
SQLAlchemy database configuration.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from rag.config import Config


if not Config.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")


engine = create_engine(
    Config.DATABASE_URL,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""

    pass
