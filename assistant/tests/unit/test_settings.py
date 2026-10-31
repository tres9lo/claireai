from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from claireai.config import Settings


def test_defaults_target_loopback_and_local_model() -> None:
    settings = Settings(_env_file=None)
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_model
    assert settings.allow_non_loopback_ollama is False


def test_non_loopback_runtime_is_rejected_by_default() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, ollama_base_url="http://192.168.1.20:11434")
    assert "non-loopback" in str(excinfo.value)


def test_non_loopback_runtime_allowed_only_with_explicit_flag() -> None:
    settings = Settings(
        _env_file=None,
        ollama_base_url="http://192.168.1.20:11434",
        allow_non_loopback_ollama=True,
    )
    assert settings.ollama_base_url == "http://192.168.1.20:11434"


def test_temperature_defaults_are_not_zero_and_differ_per_phase() -> None:
    settings = Settings(_env_file=None)
    assert settings.temperature == 0.8
    assert settings.temperature_tools == 0.2
    # Deciding on a tool must be more repeatable than phrasing the answer, and 0
    # is never the default: measured, a pinned 0 made a smaller model emit a
    # malformed tool call on every attempt.
    assert settings.temperature_tools < settings.temperature
    assert settings.tool_temperature == settings.temperature_tools


def test_temperature_is_configurable_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_TEMPERATURE", "1.1")
    monkeypatch.setenv("CLAIREAI_TEMPERATURE_TOOLS", "0")
    settings = Settings(_env_file=None)
    assert settings.temperature == 1.1
    assert settings.temperature_tools == 0.0


@pytest.mark.parametrize("value", ["-0.1", "2.1"])
def test_temperature_outside_the_supported_range_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, temperature=float(value))


def test_trailing_slash_is_normalised() -> None:
    settings = Settings(_env_file=None, ollama_base_url="http://localhost:11434/")
    assert settings.ollama_base_url == "http://localhost:11434"


@pytest.mark.parametrize("url", ["ftp://localhost:11434", "not-a-url", "http://"])
def test_invalid_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_base_url=url)


def test_timeout_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_timeout_seconds=0.5)


def test_data_dir_override_controls_paths(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime")
    assert settings.database_path == tmp_path / "runtime" / "db" / "claireai.sqlite3"
    assert settings.log_dir == tmp_path / "runtime" / "logs"


def test_default_data_dir_is_os_application_data_location() -> None:
    settings = Settings(_env_file=None)
    # Must not be the working directory or a hard-coded home path.
    assert settings.resolved_data_dir.is_absolute()
    assert settings.resolved_data_dir.name.lower() == "claireai"


def test_ensure_directories_creates_db_and_log_dirs(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime")
    settings.ensure_directories()
    assert settings.database_path.parent.is_dir()
    assert settings.log_dir.is_dir()


def test_unknown_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_typo_setting=True)  # type: ignore[call-arg]
