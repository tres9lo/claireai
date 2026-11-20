"""Read-only inbox headers through IMAP, with no send or delete capability.

Credentials come from the OS keyring account ``email_app_password`` or the
``CLAIREAI_EMAIL_APP_PASSWORD`` environment variable. The server is selected with
``CLAIREAI_EMAIL_IMAP_HOST`` (default ``imap.gmail.com``), and the address with
``CLAIREAI_EMAIL_ADDRESS``.

Only headers are fetched, with ``BODY.PEEK`` in a read-only mailbox. Reading the
inbox must not mark messages as seen, delete, move, or send anything.
"""

from __future__ import annotations

import email
import imaplib
import os
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import Message

from claireai.secrets import stored_secret

DEFAULT_HOST = "imap.gmail.com"
DEFAULT_PORT = 993
DEFAULT_TIMEOUT = 20
MAX_MESSAGES = 20


class EmailError(Exception):
    """No credentials, no connection, or an unreadable mailbox."""


@dataclass(frozen=True)
class InboxMessage:
    sender: str
    subject: str
    date: str

    def as_line(self) -> str:
        return f"{self.date} — {self.sender}: {self.subject}"


def _header(message: Message, name: str) -> str:
    value = message.get(name, "")
    try:
        return str(make_header(decode_header(value))).strip()
    except (LookupError, UnicodeError):
        return str(value).strip()


def _port() -> int:
    raw = (
        os.environ.get("CLAIREAI_EMAIL_IMAP_PORT")
        or stored_secret("email_imap_port")
        or str(DEFAULT_PORT)
    ).strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise EmailError("CLAIREAI_EMAIL_IMAP_PORT must be a number") from exc
    if not 1 <= port <= 65535:
        raise EmailError("CLAIREAI_EMAIL_IMAP_PORT must be between 1 and 65535")
    return port


def _credentials() -> tuple[str, str, str, int]:
    address = (
        os.environ.get("CLAIREAI_EMAIL_ADDRESS")
        or stored_secret("email_address")
        or ""
    ).strip()
    password = stored_secret("email_app_password")
    host = (
        os.environ.get("CLAIREAI_EMAIL_IMAP_HOST")
        or stored_secret("email_imap_host")
        or DEFAULT_HOST
    ).strip()
    if not address:
        raise EmailError("CLAIREAI_EMAIL_ADDRESS is not set")
    if not password:
        raise EmailError(
            "no email app password is stored (keyring account: email_app_password). "
            "Run `claireai --email-setup` to store one; a Gmail App Password is "
            "required, not your normal account password."
        )
    if not host:
        raise EmailError("CLAIREAI_EMAIL_IMAP_HOST must not be empty")
    return address, password, host, _port()


def _connect(host: str, port: int) -> imaplib.IMAP4_SSL:
    try:
        return imaplib.IMAP4_SSL(host, port, timeout=DEFAULT_TIMEOUT)
    except OSError as exc:
        if "name resolution" in str(exc).lower() or "name or service" in str(exc).lower():
            raise EmailError(
                f"temporary DNS failure while connecting to {host}; check the network "
                "and DNS, then retry"
            ) from exc
        raise EmailError(f"could not connect to {host}:{port}: {exc}") from exc


def _login_error(exc: BaseException, host: str) -> str:
    message = str(exc).lower()
    if "authenticationfailed" in message or "invalid credentials" in message or "auth" in message:
        hint = (
            " For Gmail, turn on 2-Step Verification and create an App Password; "
            "the normal account password will be rejected."
            if host.endswith("gmail.com")
            else ""
        )
        return f"the mail server rejected the login.{hint}"
    return f"could not read the inbox: {exc}"


def verify_login(
    address: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Prove the credentials work without reading or changing any message."""
    selected_host = (
        host
        or os.environ.get("CLAIREAI_EMAIL_IMAP_HOST")
        or stored_secret("email_imap_host")
        or DEFAULT_HOST
    ).strip()
    selected_port = port or _port()
    selected_address = (
        address
        or os.environ.get("CLAIREAI_EMAIL_ADDRESS")
        or stored_secret("email_address")
        or ""
    ).strip()
    selected_password = password or stored_secret("email_app_password")
    if not selected_address:
        raise EmailError("CLAIREAI_EMAIL_ADDRESS is not set")
    if not selected_password:
        raise EmailError("no email app password is available")
    mailbox = _connect(selected_host, selected_port)
    try:
        mailbox.login(selected_address, selected_password)
        status, _ = mailbox.select("INBOX", readonly=True)
        if status != "OK":
            raise EmailError("the INBOX could not be opened read-only")
    except imaplib.IMAP4.error as exc:
        raise EmailError(_login_error(exc, selected_host)) from exc
    finally:
        try:
            mailbox.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


def read_inbox(*, limit: int = 10) -> list[InboxMessage]:
    """Return recent unread headers without changing any mailbox state."""
    address, password, host, port = _credentials()
    limit = max(1, min(MAX_MESSAGES, int(limit)))
    mailbox = _connect(host, port)
    try:
        mailbox.login(address, password)
        status, _ = mailbox.select("INBOX", readonly=True)
        if status != "OK":
            raise EmailError("the INBOX could not be opened read-only")
        status, data = mailbox.search(None, "UNSEEN")
        if status != "OK":
            raise EmailError("the inbox search failed")
        ids = (data[0].split() if data and data[0] else [])[-limit:]
        messages: list[InboxMessage] = []
        for uid in ids:
            status, payload = mailbox.fetch(
                uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
            )
            if status != "OK" or not payload:
                continue
            raw = next((part[1] for part in payload if isinstance(part, tuple)), b"")
            if not raw:
                continue
            parsed = email.message_from_bytes(raw)
            messages.append(
                InboxMessage(
                    sender=_header(parsed, "From"),
                    subject=_header(parsed, "Subject"),
                    date=_header(parsed, "Date"),
                )
            )
        return messages
    except imaplib.IMAP4.error as exc:
        raise EmailError(_login_error(exc, host)) from exc
    except (OSError, TimeoutError) as exc:
        raise EmailError(f"could not read the inbox: {exc}") from exc
    finally:
        try:
            mailbox.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


def render(messages: list[InboxMessage]) -> str:
    if not messages:
        return "No unread messages."
    return "Unread messages (headers only; nothing was marked as read):\n" + "\n".join(
        f"- {item.as_line()}" for item in messages
    )


__all__ = ["EmailError", "InboxMessage", "read_inbox", "render", "verify_login"]
