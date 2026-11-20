"""Guided email setup verifies first, never echoes, and stores only in keyring."""

from __future__ import annotations

import pytest

from claireai import email_setup
from claireai.email_read import EmailError


def test_setup_verifies_then_stores_without_reading_a_password(monkeypatch) -> None:
    monkeypatch.delenv("CLAIREAI_EMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("CLAIREAI_EMAIL_IMAP_HOST", raising=False)
    monkeypatch.delenv("CLAIREAI_EMAIL_IMAP_PORT", raising=False)
    answers = iter(["owner@example.test", "", ""])
    stored: list[tuple[str, str]] = []
    verified: list[tuple[str, str, str, int]] = []

    text = email_setup.setup_interactive(
        ask=lambda prompt: next(answers),
        ask_secret=lambda prompt: "app-password",
        store=lambda account, value: stored.append((account, value)),
        verify=lambda address, password, host, port: verified.append(
            (address, password, host, port)
        ),
    )

    assert verified == [
        ("owner@example.test", "app-password", "imap.gmail.com", 993)
    ]
    assert ("email_app_password", "app-password") in stored
    assert ("email_address", "owner@example.test") in stored
    assert ("email_imap_host", "imap.gmail.com") in stored
    assert ("email_imap_port", "993") in stored
    assert "app-password" not in text
    assert "keyring" in text


def test_setup_does_not_store_credentials_that_fail_verification(monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_EMAIL_ADDRESS", "owner@example.test")
    monkeypatch.setenv("CLAIREAI_EMAIL_IMAP_HOST", "imap.example.test")
    stored: list[tuple[str, str]] = []

    def reject(*args) -> None:  # noqa: ANN002
        raise EmailError("the mail server rejected the login")

    with pytest.raises(EmailError, match="rejected"):
        email_setup.setup_interactive(
            ask=lambda prompt: "",
            ask_secret=lambda prompt: "wrong-password",
            store=lambda account, value: stored.append((account, value)),
            verify=reject,
        )
    assert stored == []
