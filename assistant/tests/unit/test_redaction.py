from __future__ import annotations

import pytest

from claireai.audit.redaction import REDACTED, redact


def test_api_key_value_is_masked() -> None:
    result = redact("api_key=supersecretvalue123")
    assert "supersecretvalue123" not in result
    assert REDACTED in result


def test_password_and_token_labels_are_masked() -> None:
    for text, secret in [
        ("password: hunter2", "hunter2"),
        ("token=abc123def456", "abc123def456"),
        ("client_secret: s3cr3t-value", "s3cr3t-value"),
        ("Authorization: Bearer abcdefghijklmnop", "abcdefghijklmnop"),
    ]:
        assert secret not in redact(text), text


def test_bearer_token_without_label_is_masked() -> None:
    result = redact("sending Bearer abcdefghijklmnopqrst to the server")
    assert "abcdefghijklmnopqrst" not in result


def test_known_token_prefixes_are_masked() -> None:
    original = "key sk-proj-abcdefghijklmnop and ghp_abcdefghijklmnop"
    result = redact(original)
    assert "sk-proj-abcdefghijklmnop" not in result
    assert "ghp_abcdefghijklmnop" not in result


def test_plain_text_is_unchanged() -> None:
    assert redact("list the biggest files in Downloads") == "list the biggest files in Downloads"


def test_long_text_is_truncated_with_marker() -> None:
    result = redact("x" * 50, max_length=10)
    assert result.startswith("x" * 10)
    assert "truncated 40 chars" in result


def test_none_becomes_empty_string() -> None:
    assert redact(None) == ""  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_text_round_trips(text: str) -> None:
    assert redact(text) == text
