"""Guided, read-only email credential setup.

The password is read with ``getpass`` so it never enters shell history, tested
against the IMAP server before being stored, and written only to the OS keyring.
There is deliberately no plaintext-file fallback.
"""

from __future__ import annotations

import getpass
import os
from typing import Callable

from claireai.email_read import DEFAULT_HOST, DEFAULT_PORT, EmailError, verify_login
from claireai.secrets import store_secret


def setup_interactive(
    *,
    ask: Callable[[str], str] = input,
    ask_secret: Callable[[str], str] = getpass.getpass,
    store: Callable[[str, str], None] = store_secret,
    verify: Callable[..., None] = verify_login,
) -> str:
    """Ask for address/host/password, verify read-only access, then store it."""
    address = (os.environ.get("CLAIREAI_EMAIL_ADDRESS") or "").strip()
    if not address:
        address = ask("Email address: ").strip()
    if not address:
        raise EmailError("an email address is required")

    host = (os.environ.get("CLAIREAI_EMAIL_IMAP_HOST") or "").strip()
    if not host:
        entered = ask(f"IMAP host [{DEFAULT_HOST}]: ").strip()
        host = entered or DEFAULT_HOST

    raw_port = (os.environ.get("CLAIREAI_EMAIL_IMAP_PORT") or "").strip()
    if not raw_port:
        raw_port = ask(f"IMAP port [{DEFAULT_PORT}]: ").strip() or str(DEFAULT_PORT)
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise EmailError("the IMAP port must be a number") from exc

    password = ask_secret("App password (input is hidden): ").strip()
    if not password:
        raise EmailError("an app password is required")

    verify(address, password, host, port)
    # The address and host are not secrets, but storing them beside the password
    # is what makes `--inbox` work when the window is started from the desktop
    # menu, where a shell export is not present. Environment variables still win.
    store("email_address", address)
    store("email_imap_host", host)
    store("email_imap_port", str(port))
    store("email_app_password", password)
    return (
        f"Verified read-only IMAP access for {address} at {host}:{port}.\n"
        "Address, host, port and app password are stored in the OS keyring "
        "under the claireai service. Environment variables still override them."
    )


__all__ = ["setup_interactive"]
