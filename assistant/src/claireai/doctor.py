"""Read-only environment diagnostics.

Purpose: answer "why is this not working?" without changing anything and
without printing secrets. It never creates a database, never downloads a model,
never writes configuration — it only reads and reports.

Exit codes: 0 everything ready, 1 something is missing or not ready (e.g. the
model is not installed), 2 configuration is invalid.
"""

from __future__ import annotations

import os
import platform
import sqlite3
import sys
from dataclasses import dataclass, field
from importlib import metadata
from typing import Sequence

from pydantic import ValidationError

from claireai import __version__
from claireai.brain import build_brain
from claireai.config import Settings
from claireai.platform import detect_os, supported_host

TRACKED_PACKAGES = (
    "httpx",
    "pydantic",
    "pydantic-settings",
    "platformdirs",
    "fastapi",
    "uvicorn",
    "apscheduler",
    "pytest",
)

EXPECTED_TABLES = {
    "tasks", "task_log", "summaries", "notes", "schema_migrations",
    "conversations", "conversation_messages", "user_directives", "user_preferences",
}

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class Row:
    label: str
    value: str
    state: str = OK


@dataclass(frozen=True)
class Section:
    title: str
    rows: Sequence[Row]


@dataclass(frozen=True)
class Report:
    sections: Sequence[Section]
    warnings: Sequence[str] = field(default_factory=tuple)
    errors: Sequence[str] = field(default_factory=tuple)

    @property
    def exit_code(self) -> int:
        return 2 if self.errors else (1 if self.warnings else 0)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _platform_section(settings: Settings) -> Section:
    return Section(
        title="Environment",
        rows=(
            Row("OS", f"{platform.system()} {platform.release()} ({platform.machine()})"),
            Row("Python", f"{platform.python_version()} at {sys.executable}"),
            Row(
                "Detected host",
                f"{detect_os().family} ({detect_os().name} {detect_os().release}, {detect_os().machine})",
                OK if supported_host() else WARN,
            ),
            Row("ClaireAI package", __version__),
            Row("Data directory", str(settings.resolved_data_dir)),
            Row("Log directory", str(settings.log_dir)),
        ),
    )


def _settings_row(settings: Settings) -> Section:
    return Section(
        title="Configuration",
        rows=(
            Row("Model backend", "from-scratch transformer (CPU)"),
            Row("Checkpoint", str(settings.resolved_model_checkpoint)),
            Row(
                "Data dir exists",
                "yes" if settings.resolved_data_dir.is_dir() else "no (created on first run)",
                OK if settings.resolved_data_dir.is_dir() else WARN,
            ),
            Row("CLAIREAI_ env vars set (names only)", _env_names_only()),
        ),
    )


def _database_section(settings: Settings) -> Section:
    path = settings.database_path
    if not path.is_file():
        return Section(
            title="Database",
            rows=(Row("File", f"{path} (not created yet)", WARN),),
        )

    rows = [Row("File", str(path)), Row("Size", f"{path.stat().st_size:,} bytes")]
    try:
        # Read-only: doctor must never migrate or create anything.
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            tables = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            missing = EXPECTED_TABLES - tables
            rows.append(
                Row(
                    "Schema",
                    "all expected tables present" if not missing else f"missing {sorted(missing)}",
                    OK if not missing else FAIL,
                )
            )
            if "schema_migrations" in tables:
                applied = [row["version"] for row in conn.execute("SELECT version FROM schema_migrations")]
                rows.append(Row("Migrations applied", ", ".join(applied) or "none"))
        finally:
            conn.close()
    except sqlite3.Error as exc:
        rows.append(Row("Read", f"could not read database: {exc}", FAIL))
    return Section(title="Database", rows=rows)


def _versions_section() -> Section:
    return Section(
        title="Dependencies",
        rows=tuple(Row(name, _package_version(name)) for name in TRACKED_PACKAGES),
    )


async def _brain_section(settings: Settings) -> tuple[Section, list[str], list[str]]:
    provider = settings.brain_provider
    brain = build_brain(settings)
    try:
        health = await brain.health()  # type: ignore[attr-defined]
    finally:
        await brain.aclose()  # type: ignore[attr-defined]

    warnings: list[str] = []
    errors: list[str] = []
    scratch = provider == "scratch"
    rows = [
        Row(
            "Provider",
            "scratch (on-disk checkpoint)" if scratch else f"{provider} @ {settings.resolved_brain_base_url}",
        ),
        Row(
            "Local model available",
            "yes" if health.available else f"no - {health.detail}",
            OK if health.available else FAIL,
        ),
        Row(
            "Scratch checkpoint ready" if scratch else "Connector model ready",
            "yes" if health.model_present else f"no - {health.detail}",
            OK if health.model_present else FAIL,
        ),
        Row("Checkpoint model" if scratch else "Reported models", ", ".join(health.models) or "(none)"),
    ]
    if not health.available:
        errors.append(f"{provider} brain unavailable: {health.detail}")
    elif not health.model_present:
        where = settings.resolved_model_checkpoint if scratch else settings.brain_model
        errors.append(f"the configured model is not ready: {where}")
    return Section(title="Local scratch model" if scratch else "Brain connector", rows=tuple(rows)), warnings, errors


async def collect(settings: Settings | None = None) -> Report:
    """Build a report. Read-only: no directories, files or models are created."""
    try:
        resolved = settings or Settings()
    except ValidationError as exc:
        return Report(
            sections=(),
            errors=tuple(
                f"{'.'.join(str(part) for part in error['loc']) or 'configuration'}: {error['msg']}"
                for error in exc.errors()
            ),
        )

    brain_section, warnings, errors = await _brain_section(resolved)
    checkpoint_section, checkpoint_warnings = _checkpoint_section(resolved)
    secrets_section, secret_warnings, secret_errors = _secrets_section(resolved)
    warnings = [*warnings, *checkpoint_warnings, *secret_warnings, *_migration_warnings(resolved)]
    errors = [*errors, *secret_errors]
    sections = (
        _platform_section(resolved),
        _settings_row(resolved),
        checkpoint_section,
        _database_section(resolved),
        brain_section,
        secrets_section,
        _versions_section(),
    )
    return Report(sections=sections, warnings=tuple(warnings), errors=tuple(errors))


def _checkpoint_section(settings: Settings) -> tuple[Section, list[str]]:
    """What the installed checkpoint is, and whether its sidecar agrees with it.

    A vocabulary or tokenizer mismatch does not crash: it produces confident
    gibberish, which is the failure that cost three training runs.  The sidecar is
    where that can be seen without loading 30 MB of weights.
    """
    import json
    from pathlib import Path

    checkpoint = settings.resolved_model_checkpoint
    rows = [Row("File", str(checkpoint), OK if checkpoint.is_file() else FAIL)]
    warnings: list[str] = []
    if not checkpoint.is_file():
        warnings.append(f"the configured checkpoint is missing: {checkpoint}")
        return Section(title="Installed checkpoint", rows=tuple(rows)), warnings

    rows.append(Row("Size", f"{checkpoint.stat().st_size:,} bytes"))
    sidecar = checkpoint.with_suffix(".json")
    data: dict[str, object] = {}
    if sidecar.is_file():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            warnings.append(f"{sidecar} is not readable JSON; its metadata is unknown")
    else:
        warnings.append(f"{sidecar} is missing; vocabulary and corpus cannot be checked")

    if data:
        architecture = data.get("architecture") or {}
        width = architecture.get("width") if isinstance(architecture, dict) else None
        rows += [
            Row("Name", str(data.get("name") or "(unnamed)")),
            Row(
                "Vocabulary",
                str(
                    (architecture.get("vocab_size") or architecture.get("vocabulary"))
                    if isinstance(architecture, dict)
                    else "(unknown)"
                ),
            ),
            Row(
                "Context",
                str(architecture.get("context") if isinstance(architecture, dict) else "(unknown)"),
            ),
            Row("Width", str(width or "(unknown)")),
            Row("Steps", f"{data.get('steps')} (best {data.get('best_step')})"),
            Row("Validation loss", str(data.get("validation_loss"))),
            Row("Trained on", str(data.get("training_corpus") or "(unknown)")),
        ]
        tokenizer = str(data.get("tokenizer") or "")
        if tokenizer != "corpus-bpe-v1":
            warnings.append(f"checkpoint tokenizer {tokenizer!r} is not the supported 'corpus-bpe-v1'")
        if not data.get("tokenizer_merges"):
            warnings.append("the sidecar carries no tokenizer merges; the checkpoint cannot be loaded")
        corpus = Path(str(data.get("training_corpus") or "")).name
        data_dir = Path(__file__).resolve().parents[3] / "brain-lab" / "data"
        if corpus and data_dir.is_dir():
            newest = sorted(path.name for path in data_dir.glob("chat-*"))
            if newest and corpus not in newest:
                warnings.append(
                    f"the checkpoint was trained on {corpus}; the newest corpus here is {newest[-1]}"
                )
    return Section(title="Installed checkpoint", rows=tuple(rows)), warnings


def _secrets_section(settings: Settings) -> tuple[Section, list[str], list[str]]:
    """Where a connector key lives, and whether that place is readable by others."""
    from pathlib import Path

    from claireai.secrets import KEYRING_SERVICE, keyring_secret, loose_permissions

    provider = settings.brain_provider
    warnings: list[str] = []
    errors: list[str] = []
    rows = [Row("Connector", provider)]
    if provider == "scratch":
        rows.append(Row("Key needed", "no (from-scratch model)"))
    else:
        rows.append(Row("Key set", "yes" if settings.brain_api_key_value else "no"))
        if not settings.brain_api_key_value:
            errors.append(f"the {provider} connector is selected but no key is configured")
    env_file = Path(".env")
    if env_file.is_file():
        rows.append(Row("Settings file", f"{env_file.resolve()} (exists)"))
        problem = loose_permissions(env_file)
        if problem:
            warnings.append(problem)
    else:
        rows.append(Row("Settings file", "no .env in the working directory"))
    rows.append(
        Row("Keyring", "available" if keyring_secret("probe") is not None or _keyring_present() else "not installed")
    )
    rows.append(Row("Keyring service", KEYRING_SERVICE))
    return Section(title="Connector keys", rows=tuple(rows)), warnings, errors


def _keyring_present() -> bool:
    try:
        import keyring  # noqa: F401,PLC0415 - presence check only
    except Exception:  # noqa: BLE001
        return False
    return True


def _migration_warnings(settings: Settings) -> list[str]:
    """The database exists but its migrations table is empty or absent."""
    path = settings.database_path
    if not path.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            if "schema_migrations" not in tables:
                return ["the database exists but has no schema_migrations table"]
            applied = [row[0] for row in conn.execute("SELECT version FROM schema_migrations")]
            if not applied:
                return ["the database exists but no migration has been applied"]
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return []


def _state_tag(state: str) -> str:
    return {"ok": "", "warn": "[warn] ", "fail": "[FAIL] "}.get(state, "")


def format_text(report: Report) -> str:
    lines: list[str] = ["ClaireAI doctor - read-only diagnostics", ""]
    for section in report.sections:
        lines.append(section.title)
        labels = [f"{_state_tag(row.state)}{row.label}" for row in section.rows]
        width = max((len(label) for label in labels), default=0)
        for label, row in zip(labels, section.rows):
            lines.append(f"  {label:<{width}}  {row.value}")
        lines.append("")

    if report.errors:
        lines.append("Problems to fix:")
        lines.extend(f"  - {error}" for error in report.errors)
        lines.append("")
    if report.warnings:
        lines.append("Notes:")
        lines.extend(f"  - {warning}" for warning in report.warnings)
        lines.append("")
    if not report.errors and not report.warnings:
        lines.append("All checks passed.")
    lines.append("No secrets are printed: only non-secret settings and paths are shown.")
    return "\n".join(lines)


def report_as_dict(report: Report) -> dict[str, object]:
    return {
        "sections": [
            {"title": section.title, "rows": [row.__dict__ for row in section.rows]}
            for section in report.sections
        ],
        "warnings": list(report.warnings),
        "errors": list(report.errors),
        "exit_code": report.exit_code,
    }


def _env_names_only() -> str:
    """Names of CLAIREAI_* variables that are set, never their values."""
    names = sorted(name for name in os.environ if name.startswith("CLAIREAI_"))
    return ", ".join(names) or "(none)"
