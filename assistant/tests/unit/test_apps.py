from __future__ import annotations

from pathlib import Path

import pytest

from claireai.config import Settings
from claireai.permissions import Sandbox
from claireai.tools import apps
from claireai.tools.apps import (
    ArrangeWindowsArgs,
    ListAppsArgs,
    ListWindowsArgs,
    OpenAppArgs,
    OpenInBrowserArgs,
    arrange_windows,
    list_apps,
    list_windows,
    open_app,
    open_in_browser,
)
from claireai.tools.base import ToolContext, ToolError
from claireai.windows import WindowInfo


def _context(
    *, capabilities: tuple[str, ...] = ("apps",), dry_run: bool = False,
    windows=None, cua=None,
) -> ToolContext:
    return ToolContext(
        sandbox=Sandbox.from_settings(Settings(_env_file=None, allowed_folders=[])),
        allowed_capabilities=capabilities,
        dry_run=dry_run,
        windows=windows,
        cua=cua,
    )


def _desktop_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "applications"
    folder.mkdir()
    (folder / "firefox.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Firefox\nExec=firefox %u\n",
        encoding="utf-8",
    )
    (folder / "hidden.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Hidden\nExec=hidden\nNoDisplay=true\n",
        encoding="utf-8",
    )
    return folder


def test_list_apps_reads_visible_desktop_entries(tmp_path, monkeypatch) -> None:
    folder = _desktop_dir(tmp_path)
    monkeypatch.setattr(apps, "_application_dirs", lambda: (folder,))

    result = list_apps(_context(), ListAppsArgs(query="fire"))

    assert "Firefox" in result.text
    assert "firefox.desktop" in result.text
    assert "Hidden" not in result.text


def test_open_app_requires_the_apps_capability(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(apps, "_application_dirs", lambda: (_desktop_dir(tmp_path),))

    with pytest.raises(ToolError, match="not enabled"):
        open_app(_context(capabilities=()), OpenAppArgs(app="Firefox"))


def test_open_app_launches_an_installed_entry(tmp_path, monkeypatch) -> None:
    folder = _desktop_dir(tmp_path)
    monkeypatch.setattr(apps, "_application_dirs", lambda: (folder,))
    calls: list[list[str]] = []

    class Process:
        returncode = 0

        def wait(self, timeout=None):  # noqa: ANN001
            return 0

    def spawn(command):  # noqa: ANN001
        calls.append(command)
        return Process()

    monkeypatch.setattr(apps, "_spawn", spawn)
    result = open_app(_context(), OpenAppArgs(app="Firefox"))

    assert calls
    assert "Firefox" in result.text
    assert result.detail["app"] == "firefox.desktop"


def test_open_app_rejects_an_unknown_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(apps, "_application_dirs", lambda: (_desktop_dir(tmp_path),))

    with pytest.raises(ToolError, match="no installed desktop app"):
        open_app(_context(), OpenAppArgs(app="not an app"))


def test_open_in_browser_validates_and_dry_runs(monkeypatch) -> None:
    with pytest.raises(ToolError, match="http"):
        open_in_browser(_context(dry_run=True), OpenInBrowserArgs(url="javascript:alert(1)"))

    result = open_in_browser(
        _context(dry_run=True), OpenInBrowserArgs(url="https://example.com/news")
    )
    assert "example.com" in result.text
    assert result.text.startswith("[dry run]")


def test_open_in_browser_uses_the_default_browser(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(apps.webbrowser, "open", lambda url: calls.append(url) or True)

    result = open_in_browser(
        _context(), OpenInBrowserArgs(url="https://example.com/news")
    )

    assert calls == ["https://example.com/news"]


class FakeWindows:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.row = WindowInfo(123, "Editor", 10, 20, 800, 600, active=True)

    def list(self, query=""):
        self.calls.append(("list", query))
        return [self.row]

    def focus(self, target):
        self.calls.append(("focus", target))
        return self.row

    def move(self, target, x, y):
        self.calls.append(("move", target, x, y))
        return WindowInfo(123, "Editor", x, y, 800, 600)

    def resize(self, target, width, height):
        self.calls.append(("resize", target, width, height))
        return WindowInfo(123, "Editor", 10, 20, width, height)

    def center(self, target):
        self.calls.append(("center", target))
        return self.row

    def tile(self, target, side):
        self.calls.append(("tile", target, side))
        return self.row

    def tile_all(self):
        self.calls.append(("tile_all",))
        return 3

    def close(self):
        self.calls.append(("close",))


def test_list_windows_uses_the_injected_driver() -> None:
    fake = FakeWindows()
    result = list_windows(_context(windows=fake), ListWindowsArgs(query="edit"))

    assert "Editor" in result.text
    assert fake.calls == [("list", "edit")]


def test_arrange_windows_tiles_and_tiles_all() -> None:
    fake = FakeWindows()
    result = arrange_windows(
        _context(windows=fake),
        ArrangeWindowsArgs(action="tile_left", window="Editor"),
    )
    assert "800x600" in result.text
    assert fake.calls[-1] == ("tile", "Editor", "left")

    tiled = arrange_windows(
        _context(windows=fake), ArrangeWindowsArgs(action="tile_all")
    )
    assert "3" in tiled.text
    assert fake.calls[-1] == ("tile_all",)


def test_arrange_windows_validates_required_coordinates() -> None:
    fake = FakeWindows()
    with pytest.raises(ToolError, match="both x and y"):
        arrange_windows(
            _context(windows=fake),
            ArrangeWindowsArgs(action="move", window="Editor", x=20, y=None),
        )
    assert fake.calls == []


def test_window_tools_require_the_apps_capability() -> None:
    with pytest.raises(ToolError, match="not enabled"):
        list_windows(_context(capabilities=(), windows=FakeWindows()), ListWindowsArgs())
    with pytest.raises(ToolError, match="not enabled"):
        arrange_windows(
            _context(capabilities=(), windows=FakeWindows()),
            ArrangeWindowsArgs(action="focus", window="Editor"),
        )


class FakeCua:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def press(self, key: str) -> str:
        self.calls.append(key)
        return key


def test_arrange_windows_reorders_a_browser_tab_with_keyboard_capability() -> None:
    fake, cua = FakeWindows(), FakeCua()
    result = arrange_windows(
        _context(capabilities=("apps", "keyboard"), windows=fake, cua=cua),
        ArrangeWindowsArgs(action="tab_left", window="Editor"),
    )
    assert "Moved the active tab left" in result.text
    assert fake.calls == [("focus", "Editor")]
    assert cua.calls == ["ctrl+shift+pageup"]

    with pytest.raises(ToolError, match="not enabled"):
        arrange_windows(
            _context(capabilities=("apps",), windows=FakeWindows(), cua=FakeCua()),
            ArrangeWindowsArgs(action="tab_right", window="Editor"),
        )


def test_arrange_windows_dry_run_does_not_touch_the_desktop() -> None:
    fake = FakeWindows()
    result = arrange_windows(
        _context(windows=fake, dry_run=True),
        ArrangeWindowsArgs(action="tile_all"),
    )
    assert result.text.startswith("[dry run]")
    assert fake.calls == []
