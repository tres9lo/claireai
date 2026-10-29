from __future__ import annotations

import os
from pathlib import Path

from claireai.config import Settings
from claireai.doctor import FAIL, OK, WARN, Report, Row, Section, collect, format_text, report_as_dict
from claireai.storage import Database

# Port 9 (discard) on loopback: refuses immediately, so the test does not depend
# on whether a real Ollama happens to be running on this machine.
UNREACHABLE_URL = "http://127.0.0.1:9"


def _settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        data_dir=tmp_path / "data",
        ollama_base_url=UNREACHABLE_URL,
        ollama_model="test-model",
        ollama_timeout_seconds=2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_exit_code_prefers_errors_then_warnings() -> None:
    assert Report(sections=()).exit_code == 0
    assert Report(sections=(), warnings=("note",)).exit_code == 1
    assert Report(sections=(), errors=("broken",)).exit_code == 2
    assert Report(sections=(), warnings=("note",), errors=("broken",)).exit_code == 2


def test_format_text_lists_sections_and_flags_state() -> None:
    report = Report(
        sections=(
            Section(title="Environment", rows=(Row("OS", "Linux 6.x"),)),
            Section(title="Database", rows=(Row("Schema", "missing ['tasks']", FAIL),)),
            Section(title="Warnings", rows=(Row("Data dir exists", "no", WARN),)),
        ),
        warnings=("data dir missing",),
    )
    text = format_text(report)
    assert "Environment" in text and "Database" in text
    assert "[FAIL] Schema" in text
    assert "[warn] Data dir exists" in text
    assert "Problems to fix" not in text  # no errors, only warnings
    assert "Notes:" in text
    assert "No secrets are printed" in text


def test_report_as_dict_is_serialisable() -> None:
    report = Report(
        sections=(Section(title="S", rows=(Row("k", "v", OK),)),),
        errors=("e",),
    )
    payload = report_as_dict(report)
    assert payload["exit_code"] == 2
    assert payload["sections"][0]["rows"][0] == {"label": "k", "value": "v", "state": OK}


async def test_collect_reports_unreachable_runtime_and_missing_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = await collect(settings)

    text = format_text(report)
    assert "Runtime reachable" in text
    assert "[FAIL]" in text
    assert report.exit_code == 2
    # One actionable problem at a time: an unreachable runtime is reported, and
    # the model row still names the pull command for when it comes back.
    assert len(report.errors) == 1
    assert "unreachable" in report.errors[0]
    assert "ollama pull test-model" in text


async def test_collect_is_read_only_for_the_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await collect(settings)
    assert not settings.database_path.exists(), "doctor must not create a database"


async def test_collect_reports_migrated_schema(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    Database(settings.database_path).migrate()
    report = await collect(settings)
    schema_rows = [row for section in report.sections for row in section.rows if row.label == "Schema"]
    assert schema_rows and schema_rows[0].state == OK
    assert "0001_initial" in format_text(report)


async def test_collect_never_prints_secret_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_EXAMPLE_SECRET", "do-not-print-this-value")
    report = await collect(_settings(tmp_path))
    text = format_text(report)
    assert "do-not-print-this-value" not in text
    assert "CLAIREAI_EXAMPLE_SECRET" in text  # the name is fine, the value is not


async def test_invalid_configuration_becomes_a_report_not_a_crash(monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_OLLAMA_BASE_URL", "http://10.0.0.5:11434")
    report = await collect(None)
    assert report.exit_code == 2
    assert report.sections == ()
    assert any("non-loopback" in error for error in report.errors)
