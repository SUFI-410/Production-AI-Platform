# Password recovery — operator notes

## What changed

- Public POST `/auth/password-reset/request` accepts `email` and `turnstile_token`.
- Public POST `/auth/password-reset/confirm` accepts `token` and `password`.
- Requests return the same 202 response for active, absent, inactive and
  email-throttled accounts. Incorrect configuration returns 503 for everyone.
- Random 256-bit reset tokens expire after 30 minutes. Only SHA-256 digests are
  stored. Each account has at most one current token; a new request replaces it.
- Passwords retain the existing 12–128 character policy and Argon2 hashing.
- Token consumption, password update and `auth_version` increment happen in one
  conditional UPDATE. Concurrent use of one token has at most one winner.
- Old JWTs (including legacy versionless JWTs) stop authorizing new requests after
  reset. In-flight requests already authenticated are not retroactively cancelled.
- No automatic sign-in after reset. Documents and organizations are unchanged.
- Confirmation email contains no password or recovery token.

Security reference: [OWASP Forgot Password Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Forgot_Password_Cheat_Sheet.html).

## Configuration — never commit real credentials

Keep `PASSWORD_RESET_ENABLED=false` until delivery has been configured.
Add the following to the real backend `.env` on the server:

```dotenv
PASSWORD_RESET_ENABLED=true
PASSWORD_RESET_URL=https://www.buildwithsufyan.com/reset-password
SMTP_HOST=<your provider's authenticated SMTP host>
SMTP_PORT=465
SMTP_SECURITY=ssl
SMTP_USERNAME=<SMTP login>
SMTP_PASSWORD=<SMTP password or app password>
SMTP_FROM_EMAIL=<verified sender address>
```

Alternatively use port 587 and `SMTP_SECURITY=starttls` if supported by your
provider. Both modes verify TLS certificates; plaintext SMTP is not supported.
The reset URL must be HTTPS without credentials, query or fragment. Its host is
configuration-controlled, never taken from request headers.

Configure the sender domain's SPF/DKIM/DMARC with your provider. Verify delivery
and spam placement using a mailbox you own. Do not put SMTP secrets into the
frontend, GitHub issues, screenshots, this document or chat.

Turnstile uses the existing backend `TURNSTILE_SECRET` and frontend
`VITE_TURNSTILE_SITE_KEY`. Ensure the published frontend hostname is authorized.

## Abuse limits and proxy configuration

Limits are stored in PostgreSQL, shared across API workers and restarts:

- Request: 10 attempts per client IP per 15 minutes, before Turnstile validation.
- Email: 3 accepted requests per normalized email per hour, after verification.
- Confirmation: 20 attempts per IP per 15 minutes.

Keys are HMACs of scope/email/IP, not plaintext personal information. Expired
limit rows are opportunistically pruned. These limits do not lock login accounts.
They are not a replacement for edge-level request/body limits against volumetric
abuse. Several legitimate users on one public IP share its quota.

The endpoints use `request.client.host`, not arbitrary `X-Forwarded-For` or
`CF-Connecting-IP` headers. Verify that your Nginx/Uvicorn chain forwards the
visitor address and trusts only the actual proxy. Otherwise users may share a
proxy-wide quota. Do not set forwarded-header trust to `*` on a publicly reachable
API. Do not log request bodies, reset tokens or SMTP credentials.

## Email delivery limitations

This pilot uses FastAPI in-process BackgroundTasks, not a durable queue. Account
lookup and email delivery happen after the generic response. A process restart
can lose pending delivery. Users can request another link within the limits.
If SMTP fails, the unsent token is cleared only if it is still current. Logs
contain a generic delivery failure, never SMTP exception bodies. A confirmation
email failure does not roll back a completed password change.

Monitor delivery errors before inviting customers. Add a durable transactional
outbox/worker if recovery email needs guaranteed retries at larger scale.

## Database and deployment sequence

1. Review feature branches and pass both repositories' CI before releasing.
2. Confirm a restorable PostgreSQL backup. The existing backend deployment job
   already performs a pre-migration dump and runs `alembic upgrade head` before
   restarting the API. The new revision is `b6a204c731ef`.
3. Deploy backend before frontend; configure SMTP securely and recreate/restart
   the API with the updated environment. Do not change the JWT signing key.
4. Release frontend and perform the live rehearsal below with your own account.

The migration adds three nullable/defaulted user fields and a rate-limit table;
it does not delete documents, users or organizations. Existing sessions stay valid
until their account's first reset or normal expiry.

If email delivery has problems, disable `PASSWORD_RESET_ENABLED` and restart the
API while diagnosing. Do not roll back to code that ignores `auth_version` after
password resets: it could accept previously revoked, unexpired JWTs. Keep the
revocation check and schema in place. No automatic schema downgrade is advised.

## Tests

Run the repository's full `python -m pytest -q` in the usual configured development
environment. CI runs the full suite, migration chain, and PostgreSQL recovery tests.
`PASSWORD_RESET_TEST_DATABASE_URL` opts recovery tests into a dedicated test
PostgreSQL database; the fixture creates/drops a randomly named test schema only.
Never set that variable to production. Without it, recovery tests use isolated
SQLite and the PostgreSQL concurrency test is explicitly skipped.

Focused command:

```powershell
python -m pytest tests/test_password_reset.py tests/test_password_reset_email.py tests/test_auth_dependencies.py tests/test_security.py tests/test_auth_schemas.py tests/test_config_validation.py tests/test_database_models.py -q
python -m ruff check . --select E,F,W
git diff --check
```

The API dependency now imports the RAG engine only when it is used, keeping
authentication/recovery independent of expensive model initialization.

## Live rehearsal (your own mailbox only)

1. Keep an existing session open in a second browser. Request a reset in the first.
2. Receive the email. Open the link, enter matching new passwords, and submit.
3. Verify the old password fails, the new password works, and an API request using
   the old session receives 401. Stored documents must remain unchanged.
4. Reopen the used link: it must fail. Request two links: only the latest should
   work. Check the change-notification email and mobile portrait/landscape layout.
5. Test an unknown email: same generic response, no account disclosure. Do not
   repeatedly run this rehearsal past the email/IP quotas.

Links put the token in a fragment (`#token=...`) so it is not sent to web servers.
The frontend removes it from the URL after reading it into memory. Refreshing
that page therefore requires reopening the original email link.
