"""Start with the session, without touching anything we did not write."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from claireai import autostart


def test_install_writes_a_desktop_entry_pointing_at_this_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    target = autostart.install()

    assert target == tmp_path / "autostart" / "claireai.desktop"
    text = target.read_text(encoding="utf-8")
    assert text.startswith("[Desktop Entry]")
    assert f"Exec={sys.executable} -m claireai.desktop" in text
    assert "Terminal=false" in text
    assert autostart.status()["installed"] is True


def test_install_does_not_resolve_away_the_virtual_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert sys.executable in autostart.render()


def test_arguments_with_spaces_are_quoted(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    target = autostart.install([str(tmp_path / "a folder" / "python"), "-m", "claireai.desktop"])
    assert '"' in target.read_text(encoding="utf-8").split("Exec=")[1].splitlines()[0]


def test_uninstall_removes_only_our_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    other = tmp_path / "autostart" / "something-else.desktop"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("[Desktop Entry]\n", encoding="utf-8")

    assert autostart.uninstall() is False  # nothing installed yet
    autostart.install()
    assert autostart.uninstall() is True
    assert other.is_file()
    assert not Path(autostart.desktop_file()).exists()


def test_status_reports_the_command(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    state = autostart.status()
    assert state["installed"] is False
    assert state["command"].endswith("-m claireai.desktop")


def test_unsupported_platforms_are_refused_clearly(monkeypatch) -> None:
    monkeypatch.setattr(autostart.os, "name", "nt")
    with pytest.raises(autostart.AutostartUnsupported):
        autostart.render()
