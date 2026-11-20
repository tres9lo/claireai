"""The launcher: desktop by default, terminal on request, no browser surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from claireai import app
from claireai.app import main
from claireai.config import Settings
from claireai.preflight import Requirement


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data")


def test_arguments_are_passed_to_the_cli_unchanged(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_cli(argv):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        return 7

    monkeypatch.setattr("claireai.cli.main", fake_cli)
    assert main(["--once", "hello"]) == 7
    assert seen == [["--once", "hello"]]


def test_cli_flag_opens_the_terminal_chat(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_cli(argv):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("claireai.cli.main", fake_cli)
    assert main(["--cli"]) == 0
    assert seen == [[]]


def test_the_window_opens_by_default(monkeypatch, tmp_path) -> None:
    seen: list[list[str]] = []

    def fake_desktop(argv):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("claireai.desktop.launcher.main", fake_desktop)
    monkeypatch.setattr(
        "claireai.models.ensure_installed_checkpoint", lambda settings: None
    )
    assert main(["--data-dir", str(tmp_path)]) == 0
    assert seen == [["--data-dir", str(tmp_path)]]


def test_desktop_flag_is_explicit_but_changes_nothing(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_desktop(argv):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("claireai.desktop.launcher.main", fake_desktop)
    monkeypatch.setattr(
        "claireai.models.ensure_installed_checkpoint", lambda settings: None
    )
    assert main(["--desktop"]) == 0
    assert len(seen) == 1


def test_the_removed_browser_flags_explain_themselves(capsys) -> None:
    for flag in ("--port", "--no-browser", "--ui"):
        assert app.main([flag]) == app.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "browser UI" in err
    assert "--cli" in err


def test_setup_reports_the_host_and_nothing_is_downloaded(monkeypatch, capsys, tmp_path) -> None:
    settings = _settings(tmp_path)

    def explode(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("setup without --install must not download")

    monkeypatch.setattr(app, "install", explode)
    assert app.setup_report(settings) in {app.EXIT_OK, app.EXIT_CONFIG}
    out = capsys.readouterr().out
    assert "system" in out
    assert "python" in out
    assert "speech" in out


def test_setup_install_declining_downloads_nothing(monkeypatch, capsys, tmp_path) -> None:
    settings = _settings(tmp_path)
    missing = [
        Requirement(
            key="torch", label="local model runtime", module="torch",
            spec="torch>=2.2,<3", download="about 200 MB",
        )
    ]
    monkeypatch.setattr(app, "missing_parts", lambda data_dir: missing)
    monkeypatch.setattr(app, "ask_to_install", lambda requirements: False)

    def explode(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("declining consent must not download")

    monkeypatch.setattr(app, "install", explode)
    assert app.setup_report(settings, install_missing=True) == app.EXIT_CONFIG
    out = capsys.readouterr().out
    assert "Nothing was downloaded" in out


def test_a_missing_torch_blocks_the_window_and_says_how_to_fix_it(monkeypatch, capsys) -> None:
    missing = [
        Requirement(
            key="torch", label="local model runtime", module="torch",
            spec="torch>=2.2,<3", download="about 200 MB",
        )
    ]
    monkeypatch.setattr(app, "missing_requirements", lambda: missing)
    monkeypatch.setattr(
        "claireai.desktop.launcher.main",
        lambda argv: pytest.fail("the window must not open without its runtime"),
    )

    assert app.main([]) == app.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "local model runtime" in err
    assert "--setup --install" in err
