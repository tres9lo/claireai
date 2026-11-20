"""Read-only inbox: headers only, and the mailbox is opened read-only."""

from __future__ import annotations

import pytest

from claireai import email_read


class FakeImap:
    instances: list["FakeImap"] = []

    def __init__(self, host: str, port: int, timeout: int | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.readonly: bool | None = None
        self.specs: list[str] = []
        self.logged_out = False
        FakeImap.instances.append(self)

    def login(self, address: str, password: str) -> tuple[str, list[bytes]]:
        assert (address, password) == ("owner@example.test", "app-password")
        return "OK", [b"logged in"]

    def select(self, mailbox: str, readonly: bool = False) -> tuple[str, list[bytes]]:
        assert mailbox == "INBOX"
        self.readonly = readonly
        return "OK", [b"1"]

    def search(self, charset: str | None, criterion: str) -> tuple[str, list[bytes]]:
        assert criterion == "UNSEEN"
        return "OK", [b"1 2"]

    def fetch(self, uid: bytes, spec: str) -> tuple[str, list[tuple[bytes, bytes]]]:
        self.specs.append(spec)
        header = (
            b"From: Example Sender <sender@example.test>\r\n"
            b"Subject: =?utf-8?q?Hello_from_the_test?=\r\n"
            b"Date: Fri, 20 Nov 2026 08:00:00 +0000\r\n\r\n"
        )
        return "OK", [(b"1 (BODY[HEADER.FIELDS (FROM SUBJECT DATE)] {100}", header)]

    def logout(self) -> tuple[str, list[bytes]]:
        self.logged_out = True
        return "BYE", []


@pytest.fixture()
def configured(monkeypatch):  # noqa: ANN001, ANN202
    FakeImap.instances.clear()
    monkeypatch.setenv("CLAIREAI_EMAIL_ADDRESS", "owner@example.test")
    monkeypatch.setattr(
        email_read,
        "stored_secret",
        lambda account: "app-password" if account == "email_app_password" else None,
    )
    monkeypatch.setattr(email_read.imaplib, "IMAP4_SSL", FakeImap)
    return FakeImap


def test_keyring_address_and_host_work_without_environment(monkeypatch) -> None:  # noqa: ANN001
    FakeImap.instances.clear()
    stored = {
        "email_address": "owner@example.test",
        "email_imap_host": "imap.example.test",
        "email_imap_port": "993",
        "email_app_password": "app-password",
    }
    monkeypatch.delenv("CLAIREAI_EMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("CLAIREAI_EMAIL_IMAP_HOST", raising=False)
    monkeypatch.delenv("CLAIREAI_EMAIL_IMAP_PORT", raising=False)
    monkeypatch.setattr(email_read, "stored_secret", lambda account: stored.get(account))
    monkeypatch.setattr(email_read.imaplib, "IMAP4_SSL", FakeImap)

    assert email_read.read_inbox(limit=1)
    assert FakeImap.instances[0].host == "imap.example.test"


def test_headers_are_read_without_marking_messages(configured) -> None:  # noqa: ANN001
    messages = email_read.read_inbox(limit=5)

    assert len(messages) == 2
    assert messages[0].sender == "Example Sender <sender@example.test>"
    assert messages[0].subject == "Hello from the test"
    assert messages[0].date == "Fri, 20 Nov 2026 08:00:00 +0000"
    mailbox = configured.instances[0]
    assert mailbox.readonly is True
    assert mailbox.logged_out is True
    assert all("BODY.PEEK" in spec for spec in mailbox.specs)
    assert all("BODY[]" not in spec for spec in mailbox.specs)


def test_rendering_says_nothing_was_marked_read(configured) -> None:  # noqa: ANN001
    text = email_read.render(email_read.read_inbox(limit=5))
    assert "headers only" in text
    assert "nothing was marked as read" in text
    assert "Hello from the test" in text


def test_missing_credentials_fail_before_connecting(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("CLAIREAI_EMAIL_ADDRESS", raising=False)
    monkeypatch.setattr(email_read, "stored_secret", lambda account: None)
    with pytest.raises(email_read.EmailError, match="CLAIREAI_EMAIL_ADDRESS"):
        email_read.read_inbox()


def test_verify_login_reads_no_messages(configured) -> None:  # noqa: ANN001
    email_read.verify_login("owner@example.test", "app-password", "imap.example.test", 993)
    mailbox = configured.instances[0]
    assert mailbox.readonly is True
    assert mailbox.specs == []
    assert mailbox.logged_out is True


def test_bad_port_is_rejected_before_connecting(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("CLAIREAI_EMAIL_ADDRESS", "owner@example.test")
    monkeypatch.setenv("CLAIREAI_EMAIL_IMAP_PORT", "nonsense")
    monkeypatch.setattr(email_read, "stored_secret", lambda account: "app-password")
    with pytest.raises(email_read.EmailError, match="CLAIREAI_EMAIL_IMAP_PORT"):
        email_read.read_inbox()
