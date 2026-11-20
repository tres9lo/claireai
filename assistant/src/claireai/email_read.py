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


def _credentials() -> tuple[str, str, str]:
    address = (os.environ.get("CLAIREAI_EMAIL_ADDRESS") or "").strip()
    password = stored_secret("email_app_password")
    host = (os.environ.get("CLAIREAI_EMAIL_IMAP_HOST") or DEFAULT_HOST).strip()
    if not address:
        raise EmailError("CLAIREAI_EMAIL_ADDRESS is not set")
    if not password:
        raise EmailError("no email app password is stored (keyring: email_app_password)")
    return address, password, host


def read_inbox(*, limit: int = 10) -> list[InboxMessage]:
    """Return recent unread headers without changing any mailbox state."""
    address, password, host = _credentials()
    limit = max(1, min(MAX_MESSAGES, int(limit)))
    mailbox = imaplib.IMAP4_SSL(host, DEFAULT_PORT)
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
    except (imaplib.IMAP4.error, OSError) as exc:
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


__all__ = ["EmailError", "InboxMessage", "read_inbox", "render"]
