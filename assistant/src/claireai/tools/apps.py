"""Open installed desktop apps and show a URL in the owner's browser.

App launching and browser display are separate from the HTTP browser tools:
those *read* a page, while these make a window appear on the owner's screen.
Both paths need the ``apps`` capability and an approval. The model must name an
installed desktop entry; arbitrary commands are never accepted.
"""

from __future__ import annotations

import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import ConfigDict, Field

from claireai.cua import CuaDriver, CuaError
from claireai.permissions import PermissionDenied, check_capability
from claireai.tools.base import ToolContext, ToolError, ToolResult
from claireai.tools.files import _Args as _FileArgs
from claireai.windows import WindowError, WindowManager, format_windows


class _Args(_FileArgs):
    model_config = ConfigDict(extra="forbid")


class ListAppsArgs(_Args):
    query: str = Field(default="", max_length=120, description="Optional name filter.")


class OpenAppArgs(_Args):
    app: str = Field(
        min_length=1,
        max_length=160,
        description="Desktop id or app name returned by list_apps.",
    )


class OpenInBrowserArgs(_Args):
    url: str = Field(min_length=1, max_length=4000, description="http(s) URL to show.")


class ListWindowsArgs(_Args):
    query: str = Field(default="", max_length=120, description="Optional title filter.")


class ArrangeWindowsArgs(_Args):
    action: Literal[
        "focus", "move", "resize", "tile_left", "tile_right", "maximize", "center",
        "tile_all", "tab_left", "tab_right",
    ] = Field(description="How to arrange the open window(s) or browser tab.")
    window: str = Field(
        default="", max_length=200, description="Window title or id from list_windows."
    )
    x: int | None = Field(default=None, ge=0, le=100_000)
    y: int | None = Field(default=None, ge=0, le=100_000)
    width: int | None = Field(default=None, ge=120, le=100_000)
    height: int | None = Field(default=None, ge=80, le=100_000)


def _application_dirs() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".local/share/applications",
        Path("/usr/local/share/applications"),
        Path("/usr/share/applications"),
        home / ".local/share/flatpak/exports/share/applications",
        Path("/var/lib/flatpak/exports/share/applications"),
    )


def _read_entry(path: Path) -> dict[str, str] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("[") and line.strip() != "[Desktop Entry]":
            if values:
                break
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"Name", "Exec", "NoDisplay", "Type", "Icon"}:
            values.setdefault(key.strip(), value.strip())
    if values.get("Type", "Application") != "Application":
        return None
    if values.get("NoDisplay", "").lower() == "true":
        return None
    if not values.get("Name") or not values.get("Exec"):
        return None
    return values


def installed_apps() -> list[dict[str, str]]:
    """Installed desktop entries, de-duplicated by desktop id."""
    seen: dict[str, dict[str, str]] = {}
    for folder in _application_dirs():
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.desktop")):
            entry = _read_entry(path)
            if entry is None:
                continue
            desktop_id = path.name
            if desktop_id in seen:
                continue
            seen[desktop_id] = {
                "id": desktop_id,
                "launch_id": path.stem,
                "name": entry["Name"],
                "exec": entry["Exec"],
                "path": str(path),
            }
    return sorted(seen.values(), key=lambda item: item["name"].casefold())


def _match_app(app: str) -> dict[str, str]:
    wanted = " ".join(app.split()).casefold()
    entries = installed_apps()
    for entry in entries:
        if wanted in {
            entry["id"].casefold(),
            entry["launch_id"].casefold(),
            entry["name"].casefold(),
        }:
            return entry
    matches = [
        entry
        for entry in entries
        if wanted in entry["name"].casefold() or wanted in entry["id"].casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        names = ", ".join(item["name"] for item in matches[:5])
        raise ToolError(f"more than one installed app matches {app!r}: {names}")
    raise ToolError(f"no installed desktop app matches {app!r}; use list_apps first")


def _spawn(command: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def list_apps(context: ToolContext, arguments: ListAppsArgs) -> ToolResult:
    query = " ".join(arguments.query.split()).casefold()
    entries = [
        entry
        for entry in installed_apps()
        if not query
        or query in entry["name"].casefold()
        or query in entry["id"].casefold()
    ][:30]
    if not entries:
        raise ToolError("no installed desktop app matched that name")
    lines = "\n".join(f"{entry['name']} — {entry['id']}" for entry in entries)
    return ToolResult(
        text=f"Installed apps ({len(entries)} shown):\n{lines}",
        detail={"apps": [entry["id"] for entry in entries]},
    )


def open_app(context: ToolContext, arguments: OpenAppArgs) -> ToolResult:
    try:
        check_capability("apps", getattr(context, "allowed_capabilities", ()))
    except PermissionDenied as exc:
        raise ToolError(str(exc)) from exc
    if context.dry_run:
        return ToolResult(text=f"[dry run] would open {arguments.app!r}")
    entry = _match_app(arguments.app)
    try:
        if shutil.which("gtk-launch"):
            process = _spawn(["gtk-launch", entry["launch_id"]])
        elif shutil.which("gio"):
            process = _spawn(["gio", "launch", entry["path"]])
        else:
            raise ToolError("this desktop has neither gtk-launch nor gio")
        try:
            code = process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            code = None
        if code not in {None, 0}:
            raise ToolError(f"{entry['name']} did not start (exit {code})")
    except OSError as exc:
        raise ToolError(f"could not launch {entry['name']}: {exc}") from exc
    return ToolResult(
        text=f"Opened {entry['name']} ({entry['id']}). Use computer_control to operate it.",
        detail={"app": entry["id"], "name": entry["name"]},
    )


def _window_driver(context: ToolContext) -> tuple[object, bool]:
    driver = getattr(context, "windows", None)
    if driver is not None:
        return driver, False
    return WindowManager(), True


def list_windows(context: ToolContext, arguments: ListWindowsArgs) -> ToolResult:
    try:
        check_capability("apps", getattr(context, "allowed_capabilities", ()))
    except PermissionDenied as exc:
        raise ToolError(str(exc)) from exc
    driver, owns = _window_driver(context)
    try:
        rows = driver.list(arguments.query)
        return ToolResult(
            text=format_windows(rows),
            detail={"windows": [row.id for row in rows]},
        )
    except WindowError as exc:
        raise ToolError(str(exc)) from exc
    finally:
        if owns:
            driver.close()


def arrange_windows(context: ToolContext, arguments: ArrangeWindowsArgs) -> ToolResult:
    try:
        check_capability("apps", getattr(context, "allowed_capabilities", ()))
    except PermissionDenied as exc:
        raise ToolError(str(exc)) from exc
    if context.dry_run:
        if arguments.action == "tile_all":
            text = "[dry run] would tile all open application windows"
        else:
            text = (
                f"[dry run] would {arguments.action.replace('_', ' ')} "
                f"{arguments.window or 'the chosen window'}"
            )
        return ToolResult(text=text)
    driver, owns = _window_driver(context)
    try:
        if arguments.action == "tile_all":
            count = driver.tile_all()
            return ToolResult(text=f"Tiled {count} application window(s).")
        if not arguments.window:
            raise ToolError("choose a window from list_windows for this action")
        if arguments.action in {"tab_left", "tab_right"}:
            try:
                check_capability("keyboard", getattr(context, "allowed_capabilities", ()))
            except PermissionDenied as exc:
                raise ToolError(str(exc)) from exc
            row = driver.focus(arguments.window)
            cua = getattr(context, "cua", None) or CuaDriver()
            key = "ctrl+shift+pageup" if arguments.action == "tab_left" else "ctrl+shift+pagedown"
            try:
                cua.press(key)
            except CuaError as exc:
                raise ToolError(f"could not reorder the tab: {exc}") from exc
            return ToolResult(
                text=f"Moved the active tab {'left' if arguments.action == 'tab_left' else 'right'} in {row.title}.",
                detail={"window": row.id, "title": row.title, "key": key},
            )
        if arguments.action == "focus":
            row = driver.focus(arguments.window)
        elif arguments.action == "move":
            if arguments.x is None or arguments.y is None:
                raise ToolError("move needs both x and y")
            row = driver.move(arguments.window, arguments.x, arguments.y)
        elif arguments.action == "resize":
            if arguments.width is None or arguments.height is None:
                raise ToolError("resize needs both width and height")
            row = driver.resize(arguments.window, arguments.width, arguments.height)
        elif arguments.action == "center":
            row = driver.center(arguments.window)
        else:
            side = {
                "tile_left": "left",
                "tile_right": "right",
                "maximize": "maximize",
            }[arguments.action]
            row = driver.tile(arguments.window, side)
        return ToolResult(
            text=f"{row.title}: {row.width}x{row.height} at ({row.x}, {row.y}).",
            detail={"window": row.id, "title": row.title},
        )
    except WindowError as exc:
        raise ToolError(str(exc)) from exc
    finally:
        if owns:
            driver.close()


def open_in_browser(context: ToolContext, arguments: OpenInBrowserArgs) -> ToolResult:
    try:
        check_capability("apps", getattr(context, "allowed_capabilities", ()))
    except PermissionDenied as exc:
        raise ToolError(str(exc)) from exc
    parsed = urlparse(arguments.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("only a full http(s) URL can be shown in the browser")
    if context.dry_run:
        return ToolResult(text=f"[dry run] would open {arguments.url} in the browser")
    try:
        opened = webbrowser.open(arguments.url)
        if not opened:
            opener = shutil.which("xdg-open")
            if opener is None:
                raise ToolError("no default browser or xdg-open was found")
            _spawn([opener, arguments.url])
    except OSError as exc:
        raise ToolError(f"could not open the browser: {exc}") from exc
    return ToolResult(text=f"Opened {arguments.url} in the default browser.")


__all__ = [
    "ArrangeWindowsArgs",
    "ListAppsArgs",
    "ListWindowsArgs",
    "OpenAppArgs",
    "OpenInBrowserArgs",
    "arrange_windows",
    "installed_apps",
    "list_apps",
    "list_windows",
    "open_app",
    "open_in_browser",
]
