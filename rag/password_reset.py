"""Password recovery without storing raw recovery tokens or logging secrets."""

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import hashlib
import hmac
import logging
import secrets
import smtplib
import ssl
from urllib.parse import urlsplit

from pydantic import EmailStr, TypeAdapter
from sqlalchemy import case, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from rag.config import Config
from rag.database import SessionLocal
from rag.models import PasswordResetRateLimit, User
from rag.security import hash_password

logger = logging.getLogger(__name__)


def validate_reset_configuration() -> None:
    """Fail closed before looking up an account; never accept a Host header URL."""
    url = urlsplit(Config.PASSWORD_RESET_URL)
    if (
        not Config.PASSWORD_RESET_ENABLED
        or not Config.JWT_SECRET_KEY
        or not all((Config.SMTP_HOST, Config.SMTP_USERNAME,
                    Config.SMTP_PASSWORD, Config.SMTP_FROM_EMAIL))
        or Config.SMTP_SECURITY not in {"ssl", "starttls"}
        or not 1 <= Config.SMTP_PORT <= 65535
        or url.scheme != "https"
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError("Password recovery is not configured.")
    TypeAdapter(EmailStr).validate_python(Config.SMTP_FROM_EMAIL)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def consume_limit(
    db: Session, scope: str, value: str, limit: int, seconds: int,
) -> bool:
    """Atomic shared PostgreSQL limit, including across processes and restarts."""
    key = hmac.new(
        (Config.JWT_SECRET_KEY or "").encode(),
        f"{scope}:{value}".encode(), hashlib.sha256,
    ).hexdigest()
    now = datetime.now(timezone.utc)
    model = PasswordResetRateLimit
    expired = model.expires_at <= now
    statement = insert(model).values(
        key_hash=key, attempts=1, expires_at=now + timedelta(seconds=seconds),
    ).on_conflict_do_update(
        index_elements=[model.key_hash],
        set_={
            "attempts": case((expired, 1), else_=model.attempts + 1),
            "expires_at": case(
                (expired, now + timedelta(seconds=seconds)),
                else_=model.expires_at,
            ),
        },
    ).returning(model.attempts)
    attempts = db.scalar(statement)
    # Opportunistic removal of expired buckets keeps the table bounded in time.
    if secrets.randbelow(100) == 0:
        db.execute(delete(model).where(model.expires_at < now))
    db.commit()
    return attempts is not None and attempts <= limit


def send_email(recipient: str, subject: str, body: str) -> None:
    """Encrypted SMTP only, with certificate verification and a finite timeout."""
    message = EmailMessage()
    message["From"] = Config.SMTP_FROM_EMAIL
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    context = ssl.create_default_context()
    if Config.SMTP_SECURITY == "ssl":
        connection = smtplib.SMTP_SSL(
            Config.SMTP_HOST, Config.SMTP_PORT, timeout=10, context=context,
        )
    else:
        connection = smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=10)
    with connection as smtp:
        if Config.SMTP_SECURITY == "starttls":
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
        smtp.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD)
        smtp.send_message(message)


def deliver_reset_email(email: str) -> None:
    """Background work: response time/content must not reveal account existence.

    This pilot uses in-process background tasks, not a durable email queue.
    On delivery failure/process restart, the user can request another link.
    """
    try:
        with SessionLocal() as db:
            user = db.scalar(select(User).where(
                User.email == email, User.is_active.is_(True),
            ).with_for_update())
            if user is None:
                return
            token = secrets.token_urlsafe(32)
            digest = token_digest(token)
            user.reset_token_hash = digest
            user.reset_token_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=Config.PASSWORD_RESET_EXPIRE_MINUTES,
            )
            user_id = user.id
            db.commit()
        # The fragment is not sent to web servers or in Referer headers.
        link = f"{Config.PASSWORD_RESET_URL}#token={token}"
        try:
            send_email(email, "Reset your Invoice Preflight password", (
                "A password reset was requested for your Invoice Preflight account.\n\n"
                "Open this link within "
                f"{Config.PASSWORD_RESET_EXPIRE_MINUTES} minutes:\n"
                f"{link}\n\n"
                "This link works once. A newer reset request replaces this link.\n"
                "If you did not request this, ignore this email. "
                "Your password has not changed.\n"
            ))
        except Exception:
            # Do not erase a newer token if requests overlap.
            with SessionLocal() as db:
                db.execute(update(User).where(
                    User.id == user_id, User.reset_token_hash == digest,
                ).values(reset_token_hash=None, reset_token_expires_at=None))
                db.commit()
            raise
    except Exception:
        # SMTP exceptions can contain addresses or email bodies. No traceback.
        logger.error(
            "Password reset delivery failed; check email service configuration."
        )


def reset_password(db: Session, token: str, password: str) -> str | None:
    """Consume the token and revoke sessions in one atomic database update."""
    digest = token_digest(token)
    now = datetime.now(timezone.utc)
    eligible = (
        User.reset_token_hash == digest,
        User.reset_token_expires_at > now,
        User.is_active.is_(True),
    )
    # Avoid expensive password hashing for random/expired tokens.
    if db.scalar(select(User.id).where(*eligible)) is None:
        return None
    new_hash = hash_password(password)
    email = db.scalar(update(User).where(
        User.reset_token_hash == digest,
        User.reset_token_expires_at > datetime.now(timezone.utc),
        User.is_active.is_(True),
    ).values(
        password_hash=new_hash,
        auth_version=User.auth_version + 1,
        reset_token_hash=None,
        reset_token_expires_at=None,
    ).returning(User.email))
    db.commit()
    return email


def deliver_reset_confirmation(email: str) -> None:
    try:
        send_email(email, "Your Invoice Preflight password was changed", (
            "Your password was changed and previous login sessions were revoked.\n"
            "If you did not make this change, use Forgot password on the official "
            "Invoice Preflight website to recover your account and contact support.\n"
        ))
    except Exception:
        logger.error("Password-change confirmation email could not be delivered.")
