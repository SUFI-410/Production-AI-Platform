"""Generate a signed Invoice Preflight pilot invitation link."""

from __future__ import annotations

import argparse
from urllib.parse import urlencode

from pydantic import EmailStr, TypeAdapter

from rag.security import create_registration_invite


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an expiring pilot-workspace invitation link."
        )
    )

    parser.add_argument("--email", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument(
        "--expires-hours",
        type=int,
        default=72,
    )
    parser.add_argument(
        "--frontend-url",
        default="https://www.buildwithsufyan.com",
    )

    return parser.parse_args()


def main() -> None:
    """Validate input and print the customer activation link."""

    args = _parse_arguments()
    email = str(
        TypeAdapter(EmailStr).validate_python(
            args.email
        )
    ).strip().casefold()
    organization_name = str(args.organization).strip()

    token = create_registration_invite(
        email,
        organization_name,
        expires_hours=args.expires_hours,
    )

    query = urlencode({"invite": token})
    activation_link = (
        f"{args.frontend_url.rstrip('/')}/auth?{query}"
    )

    print("Pilot invitation created.")
    print(f"Email: {email}")
    print(f"Organization: {organization_name}")
    print(f"Expires in: {args.expires_hours} hours")
    print(f"Activation link: {activation_link}")


if __name__ == "__main__":
    main()
