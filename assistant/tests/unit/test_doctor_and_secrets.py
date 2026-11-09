"""Diagnostics and key handling: the three silent failures doctor now names.

1. A checkpoint whose vocabulary/prompt does not match its sidecar (confident
   gibberish rather than a crash).
2. A database that exists but was never migrated.
3. A connector selected with no key, or a key file readable by group/others.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from claireai.config import Settings
from claireai.doctor import _checkpoint_section, _migration_warnings, _secrets_section
from claireai.permissions import Sandbox
from claireai.secrets import keyring_secret, loose_permissions, stored_secret


# --- the prompt's folder clause --------------------------------------------


def test_home_folders_are_shown_as_tilde() -> None:
    root = Path.home() / "Downloads"
    assert Sandbox(roots=(root,)).describe() == "~/Downloads"


def test_a_folder_outside_home_stays_absolute(tmp_path: Path) -> None:
    assert Sandbox(roots=(tmp_path,)).describe() == str(tmp_path)


def test_no_folders_keeps_the_runtime_sentence() -> None:
    assert Sandbox(roots=()).describe() == "(no folders are allow-listed)"


# --- key storage ------------------------------------------------------------


def test_a_key_in_the_keyring_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(get_password=lambda service, account: f"{service}:{account}")
    monkeypatch.setitem(sys.modules, "keyring", fake)
    assert keyring_secret("brain_api_key:openai") == "claireai:brain_api_key:openai"


def test_a_missing_keyring_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(service: str, account: str) -> str:
        raise RuntimeError("locked")

    monkeypatch.setitem(sys.modules, "keyring", types.SimpleNamespace(get_password=broken))
    assert keyring_secret("anything") is None


def test_the_environment_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", types.SimpleNamespace(get_password=lambda *a: None))
    monkeypatch.setenv("CLAIREAI_BRAIN_API_KEY_OPENAI", "from-env")
    assert stored_secret("brain_api_key:openai") == "from-env"


def test_a_group_readable_key_file_is_reported(tmp_path: Path) -> None:
    key_file = tmp_path / ".env"
    key_file.write_text("CLAIREAI_BRAIN_API_KEY=x\n", encoding="utf-8")
    os.chmod(key_file, 0o644)
    problem = loose_permissions(key_file)
    assert problem and "chmod 600" in problem


def test_a_private_key_file_is_not_reported(tmp_path: Path) -> None:
    key_file = tmp_path / ".env"
    key_file.write_text("x=1\n", encoding="utf-8")
    os.chmod(key_file, 0o600)
    assert loose_permissions(key_file) is None


# --- doctor sections --------------------------------------------------------


def _settings(tmp_path: Path, **overrides) -> Settings:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not really weights")
    sidecar = checkpoint.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "name": "claire-test",
                "architecture": {"width": 224, "vocab_size": 1152, "context": 288},
                "steps": 6000,
                "best_step": 5500,
                "validation_loss": 1.38,
                "training_corpus": "brain-lab/data/chat-v4",
                "tokenizer": "corpus-bpe-v1",
                "tokenizer_merges": [[1, 2]],
            }
        ),
        encoding="utf-8",
    )
    base = {
        "_env_file": None,
        "data_dir": tmp_path / "data",
        "model_checkpoint": checkpoint,
    }
    base.update(overrides)
    return Settings(**base)


def test_doctor_reports_the_installed_checkpoint(tmp_path: Path) -> None:
    section, warnings = _checkpoint_section(_settings(tmp_path))
    values = {row.label: row.value for row in section.rows}
    assert values["Name"] == "claire-test"
    assert values["Vocabulary"] == "1152"
    assert values["Context"] == "288"
    assert not any("cannot be loaded" in warning for warning in warnings)


class _KeylessConnector:
    brain_provider = "anthropic"
    brain_api_key_value = None
    database_path = Path("/nonexistent/claireai.sqlite3")


def test_doctor_flags_a_connector_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", types.SimpleNamespace(get_password=lambda *a: None))
    _, _, errors = _secrets_section(_KeylessConnector())  # type: ignore[arg-type]
    assert any("no key is configured" in error for error in errors)


def test_doctor_flags_a_database_without_migrations(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "db").mkdir(parents=True)
    database = data / "db" / "claireai.sqlite3"
    import sqlite3

    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE placeholder (id INTEGER)")
    conn.commit()
    conn.close()

    assert any("schema_migrations" in warning for warning in _migration_warnings(_settings(tmp_path)))


def test_doctor_never_prints_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Connected:
        brain_provider = "anthropic"
        brain_api_key_value = "super-secret-value"
        database_path = Path("/nonexistent/claireai.sqlite3")

    section, _, _ = _secrets_section(_Connected())  # type: ignore[arg-type]
    rendered = json.dumps([[row.label, row.value] for row in section.rows])
    assert "super-secret-value" not in rendered
