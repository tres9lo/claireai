from __future__ import annotations

import os
from pathlib import Path

from claireai.config import Settings
from claireai.doctor import FAIL, OK, WARN, Report, Row, Section, collect, format_text, report_as_dict
from claireai.storage import Database

def _settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        data_dir=tmp_path / "data",
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


async def test_collect_reports_missing_checkpoint_and_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = await collect(settings)

    text = format_text(report)
    assert "Scratch checkpoint ready" in text
    assert "[FAIL]" in text
    assert report.exit_code == 2
    # A missing local checkpoint is an actionable setup problem.
    assert len(report.errors) == 1
    assert "scratch" in report.errors[0]


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
    monkeypatch.setenv("CLAIREAI_ALLOWED_CAPABILITIES", '["teleport"]')
    report = await collect(None)
    assert report.exit_code == 2
    assert report.sections == ()
    assert any("unknown capability" in error for error in report.errors)
