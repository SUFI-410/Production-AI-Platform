"""Recovery tests use an isolated SQL database and fake email/Turnstile only."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
import re
import uuid

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth_routes import router as auth_router
from api.dependencies import get_current_user, get_db
import api.password_reset_routes as routes
from rag.config import Config
from rag.database import Base
from rag.models import Organization, PasswordResetRateLimit, User
import rag.password_reset as service
from rag.security import create_access_token, verify_password


@pytest.fixture
def recovery(monkeypatch):
    # Optional dedicated DB enables actual PostgreSQL concurrency tests.
    url = os.getenv("PASSWORD_RESET_TEST_DATABASE_URL")
    if url:
        engine = create_engine(url)
        schema = "reset_test_" + uuid.uuid4().hex
        with engine.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        engine = engine.execution_options(schema_translate_map={None: schema})
    else:
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(service, "SessionLocal", factory)
    for name, value in {
        "JWT_SECRET_KEY": "test-secret-" * 8,
        "PASSWORD_RESET_ENABLED": True,
        "PASSWORD_RESET_URL": "https://example.com/reset-password",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_USERNAME": "test-user",
        "SMTP_PASSWORD": "not-a-real-password",
        "SMTP_FROM_EMAIL": "support@example.com",
        "SMTP_SECURITY": "ssl",
        "SMTP_PORT": 465,
    }.items():
        monkeypatch.setattr(Config, name, value)
    monkeypatch.setattr(routes, "verify_turnstile", lambda *args: None)
    sent = []
    monkeypatch.setattr(service, "send_email", lambda *args: sent.append(args))
    user_id = uuid.uuid4()
    with factory() as db:
        org = Organization(name="Recovery test")
        db.add(org)
        db.flush()
        db.add(User(
            id=user_id, organization_id=org.id, email="owner@example.com",
            password_hash=service.hash_password("OriginalPassword123!"),
        ))
        db.commit()
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(auth_router)

    def database():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    yield TestClient(app), factory, sent, user_id
    if url:
        with engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
    engine.dispose()


def request_link(client, email="owner@example.com"):
    return client.post("/auth/password-reset/request", json={
        "email": email, "turnstile_token": "test-verification",
    })


def extract_token(sent):
    return re.search(r"#token=([A-Za-z0-9_-]{43})", sent[-1][2]).group(1)


def confirm(client, token, password="ReplacementPassword123!"):
    return client.post("/auth/password-reset/confirm", json={
        "token": token, "password": password,
    })


def test_complete_recovery_revokes_sessions_and_preserves_tenant(recovery):
    client, factory, sent, user_id = recovery
    old_token = create_access_token(str(user_id))
    response = request_link(client, "Owner@Example.com")
    assert response.status_code == 202
    assert response.headers["cache-control"] == "no-store"
    token = extract_token(sent)
    with factory() as db:
        user = db.get(User, user_id)
        org_id = user.organization_id
        assert user.reset_token_hash == service.token_digest(token)
        assert user.reset_token_hash != token
    assert confirm(client, token).status_code == 200
    assert len(sent) == 2
    assert token not in sent[-1][2]
    assert "ReplacementPassword123!" not in sent[-1][2]
    assert confirm(client, token).status_code == 400
    with factory() as db:
        user = db.get(User, user_id)
        assert user.organization_id == org_id
        assert user.auth_version == 1
        assert user.reset_token_hash is None
        assert verify_password("ReplacementPassword123!", user.password_hash)
        assert not verify_password("OriginalPassword123!", user.password_hash)
        with pytest.raises(HTTPException) as error:
            get_current_user(
                HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=old_token,
                ), db,
            )
        assert error.value.status_code == 401
    assert client.post("/auth/login", json={
        "email": "owner@example.com", "password": "OriginalPassword123!",
    }).status_code == 401
    login = client.post("/auth/login", json={
        "email": "owner@example.com", "password": "ReplacementPassword123!",
    })
    assert login.status_code == 200
    assert client.get("/auth/me", headers={
        "Authorization": "Bearer " + login.json()["access_token"],
    }).status_code == 200


def test_unknown_and_inactive_emails_have_identical_response(recovery):
    client, factory, sent, user_id = recovery
    existing = request_link(client)
    missing = request_link(client, "missing@example.com")
    with factory() as db:
        db.execute(update(User).where(User.id == user_id).values(is_active=False))
        db.commit()
    inactive = request_link(client)
    assert existing.status_code == missing.status_code == inactive.status_code == 202
    assert existing.json() == missing.json() == inactive.json()
    assert len(sent) == 1


def test_new_request_invalidates_old_link(recovery):
    client, _, sent, _ = recovery
    request_link(client)
    old = extract_token(sent)
    request_link(client)
    new = extract_token(sent)
    assert new != old
    assert confirm(client, old).status_code == 400
    assert confirm(client, new).status_code == 200


@pytest.mark.parametrize("kind", ["expired", "inactive", "random"])
def test_invalid_links_do_not_change_password(recovery, kind):
    client, factory, sent, user_id = recovery
    request_link(client)
    token = extract_token(sent)
    with factory() as db:
        user = db.get(User, user_id)
        if kind == "expired":
            user.reset_token_expires_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            )
        if kind == "inactive":
            user.is_active = False
        db.commit()
    if kind == "random":
        token = "a" * 43
    assert confirm(client, token).status_code == 400
    with factory() as db:
        user = db.get(User, user_id)
        assert user.auth_version == 0
        assert verify_password("OriginalPassword123!", user.password_hash)


@pytest.mark.parametrize("password", ["short", "x" * 129])
def test_password_policy_does_not_consume_link(recovery, password):
    client, _, sent, _ = recovery
    request_link(client)
    token = extract_token(sent)
    assert confirm(client, token, password).status_code == 422
    assert confirm(client, token).status_code == 200


def test_invitation_and_access_tokens_cannot_reset_password(recovery):
    client, _, _, user_id = recovery
    from rag.security import create_registration_invite
    for token in (
        create_access_token(str(user_id)),
        create_registration_invite("owner@example.com", "Test organization"),
    ):
        assert confirm(client, token).status_code == 422


def test_email_limit_is_shared_and_generic(recovery):
    client, factory, sent, _ = recovery
    bodies = [request_link(client).json() for _ in range(4)]
    assert all(body == bodies[0] for body in bodies)
    assert len(sent) == 3
    with factory() as db:
        keys = db.scalars(select(PasswordResetRateLimit.key_hash)).all()
        assert all(len(key) == 64 and "owner" not in key for key in keys)


def test_ip_limit_and_expiration(recovery):
    client, factory, _, _ = recovery
    for _ in range(10):
        assert request_link(client, "missing@example.com").status_code == 202
    assert request_link(client).status_code == 429
    with factory() as db:
        db.execute(update(PasswordResetRateLimit).values(
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        ))
        db.commit()
    assert request_link(client).status_code == 202


def test_confirm_rate_limit(recovery):
    client, _, _, _ = recovery
    for _ in range(20):
        assert confirm(client, "a" * 43).status_code == 400
    assert confirm(client, "a" * 43).status_code == 429


def test_verification_failure_sends_nothing(recovery, monkeypatch):
    client, _, sent, _ = recovery

    def reject(*args):
        raise HTTPException(403, "Human verification failed.")

    monkeypatch.setattr(routes, "verify_turnstile", reject)
    assert request_link(client).status_code == 403
    assert not sent


def test_disabled_recovery_is_explicit_and_does_not_touch_account(
    recovery, monkeypatch,
):
    client, _, sent, _ = recovery
    monkeypatch.setattr(Config, "PASSWORD_RESET_ENABLED", False)
    assert request_link(client).status_code == 503
    assert not sent


@pytest.mark.parametrize("name,value", [
    ("SMTP_SECURITY", "none"), ("SMTP_PASSWORD", ""),
    ("SMTP_FROM_EMAIL", "invalid"), ("SMTP_PORT", 0),
    ("PASSWORD_RESET_URL", "http://example.com/reset-password"),
    ("PASSWORD_RESET_URL", "https://example.com/reset-password?redirect=evil"),
    ("PASSWORD_RESET_URL", "https://name:secret@example.com/reset-password"),
])
def test_unsafe_email_configuration_rejected(recovery, monkeypatch, name, value):
    monkeypatch.setattr(Config, name, value)
    with pytest.raises(ValueError):
        service.validate_reset_configuration()


def test_delivery_failure_is_generic_and_invalidates_unsent_token(
    recovery, monkeypatch, caplog,
):
    client, factory, _, user_id = recovery

    def fail(*args):
        raise RuntimeError("secret-token-and-email-body")

    monkeypatch.setattr(service, "send_email", fail)
    assert request_link(client).status_code == 202
    with factory() as db:
        assert db.get(User, user_id).reset_token_hash is None
    assert "delivery failed" in caplog.text
    assert "secret-token-and-email-body" not in caplog.text


def test_confirmation_delivery_failure_does_not_undo_password_change(
    recovery, monkeypatch,
):
    client, factory, sent, user_id = recovery
    request_link(client)
    token = extract_token(sent)

    def fail(*args):
        raise RuntimeError("smtp unavailable")

    monkeypatch.setattr(service, "send_email", fail)
    assert confirm(client, token).status_code == 200
    with factory() as db:
        assert db.get(User, user_id).auth_version == 1


@pytest.mark.skipif(
    not os.getenv("PASSWORD_RESET_TEST_DATABASE_URL"),
    reason="Requires an explicit disposable PostgreSQL test database",
)
def test_postgres_concurrent_consumption_has_one_winner(recovery):
    client, factory, sent, _ = recovery
    request_link(client)
    token = extract_token(sent)

    def consume(_):
        with factory() as db:
            return service.reset_password(db, token, "ConcurrentPassword123!")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(consume, range(2)))
    assert results.count("owner@example.com") == 1
    assert results.count(None) == 1
