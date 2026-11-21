"""`claireai-desktop` must explain a missing toolkit, not traceback.

The window modules import tkinter and customtkinter at module level, so these
tests never import them: the launcher's whole job is to decide before that
happens. Missing modules are simulated through ``sys.modules``.
"""

from __future__ import annotations

import sys
import tomllib
import types
from pathlib import Path

from claireai import app as app_module
from claireai.desktop import launcher


def _present(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tkinter", types.ModuleType("tkinter"))
    monkeypatch.setitem(sys.modules, "customtkinter", types.ModuleType("customtkinter"))


def test_missing_customtkinter_points_at_the_extra(monkeypatch, capsys) -> None:
    _present(monkeypatch)
    monkeypatch.setitem(sys.modules, "customtkinter", None)

    assert launcher.main([]) == 2

    error = capsys.readouterr().err
    assert 'pip install "claireai[desktop]"' in error
    assert "Traceback" not in error


def test_missing_tk_names_the_os_package(monkeypatch, capsys) -> None:
    _present(monkeypatch)
    monkeypatch.setitem(sys.modules, "tkinter", None)

    assert launcher.main([]) == 2

    error = capsys.readouterr().err
    assert "python3-tk" in error
    assert "Traceback" not in error


def test_a_complete_install_hands_over_to_the_window(monkeypatch) -> None:
    _present(monkeypatch)
    seen: list[list[str]] = []
    window = types.ModuleType("claireai.desktop.app")

    def fake_main(argv: list[str]) -> int:
        seen.append(argv)
        return 7

    window.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claireai.desktop.app", window)

    assert launcher.main(["--data-dir", "/tmp/elsewhere"]) == 7
    assert seen == [["--data-dir", "/tmp/elsewhere"]]


def test_claireai_desktop_also_goes_through_the_launcher(monkeypatch, tmp_path) -> None:
    seen: list[list[str]] = []
    target = tmp_path / "elsewhere"
    module = types.ModuleType("claireai.desktop.launcher")

    def fake_main(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claireai.desktop.launcher", module)
    # Routing test: CI does not install the optional torch runtime.
    monkeypatch.setattr(app_module, "missing_requirements", lambda: [])

    assert app_module.main(["--desktop", "--data-dir", str(target)]) == 0
    assert seen == [["--data-dir", str(target)]]


def test_the_console_script_is_wired_to_the_launcher() -> None:
    config = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert config["project"]["scripts"]["claireai-desktop"] == "claireai.desktop.launcher:main"
