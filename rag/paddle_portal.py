"""Create temporary Paddle portal sessions for authenticated tenants."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx
from rag.config import Config
from rag.models import Subscription
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


class PaddlePortalError(RuntimeError):
    """Portal creation failed."""


class PaddlePortalUnavailableError(PaddlePortalError):
    """The organization has no linked Paddle subscription."""


class PaddlePortalConfigurationError(PaddlePortalError):
    """Paddle is not configured."""


class PaddlePortalAPIError(PaddlePortalError):
    """Paddle could not create a valid portal session."""


class PaddlePortalService:
    """Resolve customer identity from the database, never from browser input."""

    def __init__(self, db: Session, *, client: httpx.Client | None = None):
        self.db = db
        self.client = client

    def create_session(self, organization_id: UUID) -> str:
        try:
            subscription = self.db.scalar(
                select(Subscription).where(
                    Subscription.organization_id == organization_id
                )
            )
        except SQLAlchemyError:
            raise PaddlePortalError("Unable to resolve billing account.") from None

        if (
            subscription is None
            or subscription.provider != "paddle"
            or not re.fullmatch(
                r"ctm_[a-z0-9]{26}", subscription.provider_customer_id or ""
            )
            or not re.fullmatch(
                r"sub_[a-z0-9]{26}", subscription.provider_subscription_id or ""
            )
        ):
            raise PaddlePortalUnavailableError("No linked Paddle subscription.")

        environment = Config.PADDLE_ENVIRONMENT.strip().lower()
        hosts = {
            "sandbox": ("sandbox-api.paddle.com", "sandbox-customer-portal.paddle.com"),
            "live": ("api.paddle.com", "customer-portal.paddle.com"),
        }
        api_key = Config.PADDLE_API_KEY.strip()
        if environment not in hosts or not api_key:
            raise PaddlePortalConfigurationError("Paddle is not configured.")
        api_host, portal_host = hosts[environment]
        customer_id = subscription.provider_customer_id
        subscription_id = subscription.provider_subscription_id
        client = self.client
        owns_client = client is None
        if client is None:
            client = httpx.Client(timeout=Config.REQUEST_TIMEOUT)
        try:
            response = client.post(
                f"https://{api_host}/customers/{customer_id}/portal-sessions",
                headers={"Authorization": f"Bearer {api_key}", "Paddle-Version": "1"},
                json={"subscription_ids": [subscription_id]},
            )
            response.raise_for_status()
            data = response.json()["data"]
            if data["customer_id"] != customer_id:
                raise ValueError("Customer mismatch")
            url = data["urls"]["general"]["overview"]
            if not isinstance(url, str) or any(c.isspace() for c in url):
                raise ValueError("Invalid portal URL")
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.netloc != portal_host
                or not parsed.path.startswith("/cpl_")
                or not parse_qs(parsed.query).get("token", [""])[0]
            ):
                raise ValueError("Invalid portal URL")
            return url
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise PaddlePortalAPIError(
                "Unable to create Paddle portal session."
            ) from None
        finally:
            if owns_client:
                client.close()
