"""SMTP transport tests; no external connections or real credentials."""

from unittest.mock import MagicMock

import pytest

from rag.config import Config
import rag.password_reset as service


@pytest.mark.parametrize("security,port", [("ssl", 465), ("starttls", 587)])
def test_smtp_is_encrypted_and_authenticated(monkeypatch, security, port):
    monkeypatch.setattr(Config, "SMTP_SECURITY", security)
    monkeypatch.setattr(Config, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(Config, "SMTP_PORT", port)
    monkeypatch.setattr(Config, "SMTP_USERNAME", "sender@example.com")
    monkeypatch.setattr(Config, "SMTP_PASSWORD", "fake-secret")
    monkeypatch.setattr(Config, "SMTP_FROM_EMAIL", "sender@example.com")
    constructor = MagicMock()
    smtp = constructor.return_value.__enter__.return_value
    monkeypatch.setattr(
        service.smtplib, "SMTP_SSL" if security == "ssl" else "SMTP", constructor,
    )
    service.send_email("owner@example.com", "Recovery", "Test email body")
    assert constructor.call_args.kwargs["timeout"] == 10
    smtp.login.assert_called_once_with("sender@example.com", "fake-secret")
    if security == "ssl":
        assert constructor.call_args.kwargs["context"].check_hostname
    else:
        assert smtp.starttls.call_args.kwargs["context"].check_hostname
        methods = [call[0] for call in smtp.method_calls]
        assert methods.index("starttls") < methods.index("login")
    message = smtp.send_message.call_args.args[0]
    assert message["To"] == "owner@example.com"
    assert message["From"] == "sender@example.com"
    assert "fake-secret" not in str(message)
