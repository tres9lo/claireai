"""One command to run ClaireAI.

Everything the app needs is behind this module:

    claireai            # start the local app and open it
    claireai --cli      # the terminal chat, no browser
    claireai --once "..."   # any existing flag still works (delegated to cli)

It prepares the data directory, migrates the database, binds the loopback
server, prints where it is, opens a browser, and stays in the foreground until
Ctrl+C. Nothing here is privileged: no sudo, no LAN listener, no daemon that
outlives the terminal.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from claireai.config import Settings
from claireai.presence import PresenceStream
from claireai.presence_server import build_server
from claireai.runtime_control import RuntimeControl
from claireai.storage import Database

EXIT_OK = 0
EXIT_CONFIG = 2
DEFAULT_PORT = 8765
PORT_ATTEMPTS = 20

HTML_PATH = Path(__file__).resolve().parents[3] / "desktop" / "presence" / "index.html"


@dataclass(frozen=True)
class Running:
    """What the launcher resolved, so the banner and tests agree."""

    url: str
    host: str
    port: int
    data_dir: Path
    database_path: Path
    browser_tools: bool


def pick_port(host: str, preferred: int, *, attempts: int = PORT_ATTEMPTS) -> int:
    """First free port at or after ``preferred``, probing only loopback ports.

    Trying the bind is the only honest test: another process may hold the port
    without appearing in any file we could read.
    """
    import socket

    for candidate in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise OSError(f"no free port for {host} in {preferred}..{preferred + attempts - 1}")


def prepare(
    settings: Settings,
    *,
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    html_path: Path | None = None,
) -> tuple[object, Running]:
    """Migrate, bind and describe the app. Does not serve yet."""
    settings.ensure_directories()
    database_path = settings.database_path
    Database(database_path).migrate()

    page = html_path or HTML_PATH
    if not page.is_file():
        raise FileNotFoundError(f"the app page is missing: {page}")

    chosen = pick_port(host, port)
    server = build_server(
        PresenceStream(),
        html_path=page,
        host=host,
        port=chosen,
        database_path=database_path,
        control_path=settings.control_state_path,
    )
    bound_host, bound_port = server.server_address[:2]
    return server, Running(
        url=f"http://{bound_host}:{bound_port}/",
        host=str(bound_host),
        port=int(bound_port),
        data_dir=settings.resolved_data_dir,
        database_path=database_path,
        browser_tools=bool(RuntimeControl(settings.control_state_path).read()["browser"]),
    )


def banner(running: Running, *, opening: bool = True) -> str:
    """The whole point: say where it is and how to stop it, in a few lines."""
    tools = "on (approvals still required)" if running.browser_tools else "off — enable it in the panel"
    where = (
        "Opening it in your browser. Press Ctrl+C to stop."
        if opening
        else f"Open {running.url} yourself. Press Ctrl+C to stop."
    )
    return "\n".join(
        [
            "",
            "ClaireAI is running.",
            f"  app       {running.url}",
            f"  summary   {running.url}today   (read-only page)",
            f"  data      {running.data_dir}",
            f"  web tools {tools}",
            "",
            where,
            "",
        ]
    )


def serve(server) -> None:  # noqa: ANN001
    """Block until Ctrl+C, then close the listener."""
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def wants_browser(*, flag: bool, stream=None) -> bool:  # noqa: ANN001
    """Open a browser only for a human at a terminal, and only if allowed."""
    if not flag:
        return False
    if os.environ.get("CLAIREAI_NO_BROWSER", "").strip().lower() in {"1", "true", "yes"}:
        return False
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


# Flags this launcher owns. Anything else is a CLI flag and is passed through
# untouched, which is what keeps `claireai --once "..."` working after the
# default behaviour became "start the app".
APP_FLAGS = frozenset(
    {
        "--cli", "-i", "--port", "--data-dir", "--no-browser", "-h", "--help",
        "--desktop", "--install-autostart", "--uninstall-autostart", "--autostart-status",
    }
)


def main(
    argv: Sequence[str] | None = None,
    *,
    open_browser: Callable[[str], bool] | None = None,
    serve_forever: Callable[[object], None] | None = None,
) -> int:
    """Launch the app, or hand the arguments to the CLI unchanged."""
    args = list(sys.argv[1:] if argv is None else argv)
    first = args[0].split("=", 1)[0] if args else ""

    if first in {"--cli", "-i"}:  # the terminal chat, no server
        from claireai.cli import main as cli_main

        return cli_main([])
    if args and first not in APP_FLAGS:
        from claireai.cli import main as cli_main

        return cli_main(args)

    import argparse

    parser = argparse.ArgumentParser(
        prog="claireai", description="Start the local ClaireAI app (loopback only)."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"first port to try (default {DEFAULT_PORT})")
    parser.add_argument("--data-dir", help="data directory (default: the OS application-data directory)")
    parser.add_argument("--no-browser", action="store_true", help="print the address instead of opening it")
    parser.add_argument("--desktop", action="store_true", help="open the native window instead of the web UI")
    parser.add_argument("--install-autostart", action="store_true", help="start ClaireAI with the desktop session")
    parser.add_argument("--uninstall-autostart", action="store_true", help="stop starting it with the session")
    parser.add_argument("--autostart-status", action="store_true", help="report the autostart entry")
    parsed = parser.parse_args(args)

    from claireai import autostart

    if parsed.autostart_status:
        state = autostart.status()
        print(f"autostart {'installed' if state['installed'] else 'not installed'}: {state['path']}")
        print(f"command: {state['command']}")
        return EXIT_OK
    if parsed.install_autostart:
        try:
            target = autostart.install()
        except autostart.AutostartUnsupported as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        print(f"ClaireAI will start with the desktop session: {target}")
        print("Remove it later with: claireai --uninstall-autostart")
        return EXIT_OK
    if parsed.uninstall_autostart:
        removed = autostart.uninstall()
        print("autostart entry removed" if removed else "no autostart entry was installed")
        return EXIT_OK
    if parsed.desktop:
        from claireai.desktop.app import main as desktop_main

        return desktop_main(["--data-dir", parsed.data_dir] if parsed.data_dir else [])

    # Only override the directory when asked: Settings() reads CLAIREAI_DATA_DIR.
    settings = (
        Settings(data_dir=Path(parsed.data_dir).expanduser())
        if parsed.data_dir
        else Settings()
    )
    try:
        server, running = prepare(settings, port=parsed.port)
    except (OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    opening = wants_browser(flag=not parsed.no_browser)
    print(banner(running, opening=opening), flush=True)
    if opening:
        (open_browser or webbrowser.open)(running.url)
    (serve_forever or serve)(server)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
