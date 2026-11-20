"""The menu entry runs this interpreter, not whichever claireai is on PATH."""

from __future__ import annotations

import sys

from claireai import menu


def test_the_entry_runs_this_python_and_uses_the_brand_icon(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    target = menu.install()

    text = target.read_text(encoding="utf-8")
    assert target == tmp_path / "applications" / "claireai.desktop"
    assert f"Exec={sys.executable} -m claireai.app" in text
    assert "Icon=claireai" in text
    assert "StartupWMClass=Claireai" in text
    assert menu.icon_path().is_file()
    assert menu.icon_path().read_bytes().startswith(b"\x89PNG")


def test_install_is_reversible(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    menu.install()

    assert menu.uninstall() is True
    assert menu.uninstall() is False
    assert not menu.entry_path().exists()
    assert not menu.icon_path().exists()


def test_a_path_with_spaces_is_quoted() -> None:
    text = menu.render(["/opt/My Apps/python", "-m", "claireai.app"])

    assert 'Exec="/opt/My Apps/python" -m claireai.app' in text
