from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Config


def test_validate_api_accepts_complete_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chroma_dir = tmp_path / "chroma"
    log_dir = tmp_path / "logs"

    monkeypatch.setattr(
        Config,
        "OPENAI_API_KEY",
        "test-openai-key",
    )
    monkeypatch.setattr(
        Config,
        "DATABASE_URL",
        "postgresql+psycopg://test",
    )
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        "x" * 64,
    )
    monkeypatch.setattr(
        Config,
        "CHROMA_DIR",
        chroma_dir,
    )
    monkeypatch.setattr(
        Config,
        "LOG_DIR",
        log_dir,
    )

    Config.validate_api()

    assert chroma_dir.is_dir()
    assert log_dir.is_dir()


def test_validate_api_rejects_missing_jwt_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        Config,
        "OPENAI_API_KEY",
        "test-openai-key",
    )
    monkeypatch.setattr(
        Config,
        "DATABASE_URL",
        "postgresql+psycopg://test",
    )
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        None,
    )
    monkeypatch.setattr(
        Config,
        "CHROMA_DIR",
        tmp_path / "chroma",
    )
    monkeypatch.setattr(
        Config,
        "LOG_DIR",
        tmp_path / "logs",
    )

    with pytest.raises(
        RuntimeError,
        match="JWT_SECRET_KEY",
    ):
        Config.validate_api()


def test_validate_api_reports_all_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Config,
        "OPENAI_API_KEY",
        None,
    )
    monkeypatch.setattr(
        Config,
        "DATABASE_URL",
        None,
    )
    monkeypatch.setattr(
        Config,
        "JWT_SECRET_KEY",
        None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        Config.validate_api()

    message = str(exc_info.value)

    assert "OPENAI_API_KEY" in message
    assert "DATABASE_URL" in message
    assert "JWT_SECRET_KEY" in message
