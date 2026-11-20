#!/usr/bin/env python3
"""ClaireAI installer: detect the host, show the download list, then ask.

Run it from a checkout:

    ./install.sh              # Linux / macOS
    .\\install.ps1             # Windows PowerShell

It refuses an unsupported Python, explains a missing Tk toolkit instead of
failing halfway, creates one virtual environment, installs the complete desktop
application (window, local model runtime and voice input), and offers the local
speech-to-text model. Nothing is downloaded before the list is printed and the
answer is yes; ``--yes`` is the explicit non-interactive form of that consent.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assistant" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from claireai.preflight import (  # noqa: E402 - after the checkout is on sys.path
    SPEECH_MODEL_DOWNLOAD,
    cpu_index_url,
    detect_environment,
    format_environment,
    format_plan,
    install_speech_model,
    requirements_for,
    speech_model_installed,
)

EXIT_CONFIG = 2

TK_HINTS = {
    "linux": "install python3-tk (Debian/Ubuntu), python3-tkinter (Fedora) or tk (Arch), then run this again",
    "macos": "use the python.org installer, which includes Tk, or `brew install python-tk`",
    "windows": "repair the Python installation and keep the tcl/tk feature selected",
    "other": "install Tk for this Python, then run this again",
}


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


TK_PROBE = (
    "import tkinter as tk;"
    "r=tk.Tk();"
    "print(r.tk.call('::tk::pkgconfig','get','fontsystem'));"
    "r.destroy()"
)


def tk_font_system(python: str | Path) -> str:
    """Whether this interpreter's Tk can draw antialiased fonts.

    Some conda builds ship Tk 9 with the core X11 font system only: every glyph
    comes from a bitmap font and the whole window reads as pixelated. Tk reports
    ``xft`` when it has fontconfig, so that is the check worth making before
    choosing the interpreter that will run the window.
    """
    if os.name != "posix" or sys.platform != "linux":
        return "native"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no-display"
    try:
        result = subprocess.run(
            [str(python), "-c", TK_PROBE], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _probe_python(candidate: str) -> tuple[str, str] | None:
    """Return (path, tk backend) for a Python that could run the window."""
    path = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
    if not path or not Path(path).is_file():
        return None
    try:
        version = subprocess.run(
            [path, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=10,
        )
        tk = subprocess.run(
            [path, "-c", "import tkinter"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if version.returncode != 0 or tk.returncode != 0:
        return None
    major, minor = (int(part) for part in version.stdout.strip().split("."))
    if (major, minor) < (3, 11):
        return None
    return path, tk_font_system(path)


def choose_base_python(environment, requested: str | None) -> tuple[str, str]:
    """Prefer a Python whose Tk can render fonts, not just one that imports Tk."""
    if requested:
        probe = _probe_python(requested)
        if probe is None:
            raise SystemExit(f"ERROR: {requested} is not a Python 3.11+ with tkinter")
        return probe[0], "requested with --python"

    candidates = [sys.executable, "/usr/bin/python3", "python3.12", "python3.11", "python3"]
    seen: set[str] = set()
    usable: list[tuple[str, str]] = []
    for candidate in candidates:
        probe = _probe_python(candidate)
        if probe is None or probe[0] in seen:
            continue
        seen.add(probe[0])
        usable.append(probe)
    for path, backend in usable:
        if backend in {"xft", "native", "no-display"}:
            return path, f"Tk font backend: {backend}"
    if usable:
        path, backend = usable[0]
        print(
            f"\nWARNING: every Python here uses Tk's {backend} font system; text will "
            "look pixelated. Install python3-tk/fontconfig or use a Python with Xft Tk."
        )
        return path, f"Tk font backend: {backend}"
    raise SystemExit("ERROR: no Python 3.11+ with tkinter was found.")


def stream(command: list[str], *, cwd: Path = ROOT) -> int:
    """Run a command with its output visible, which is where sizes appear."""
    print(f"\n$ {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\n"), flush=True)
    return process.wait()


def install_command(python: str, spec: str, *, index_url: str = "") -> list[str]:
    """Prefer uv when it is present; both receive the same requirements."""
    uv = shutil.which("uv")
    if uv:
        command = [uv, "pip", "install", "--python", python, spec]
    else:
        command = [python, "-m", "pip", "install", spec]
    if index_url:
        command += ["--index-url", index_url]
    return command


def ask(question: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("\nNo terminal to ask in. Nothing was downloaded; re-run with --yes to approve the list.")
        return False
    return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes"}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="install.sh",
        description="Install ClaireAI's desktop app from this checkout, with consent.",
    )
    parser.add_argument("--yes", action="store_true", help="approve the printed download list")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen, then stop")
    parser.add_argument(
        "--venv",
        default=str(ROOT / "assistant" / ".venv"),
        help="virtual environment to create or reuse (default: assistant/.venv)",
    )
    parser.add_argument(
        "--no-speech-model",
        action="store_true",
        help="skip the optional ~75 MB speech-to-text model",
    )
    parser.add_argument(
        "--python",
        help="base interpreter for the virtual environment (default: the first with a usable Tk)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environment = detect_environment()
    venv = Path(args.venv).expanduser().resolve()
    python = venv_python(venv)
    base_python, base_reason = choose_base_python(environment, args.python)

    print("ClaireAI installer")
    print(format_environment(environment))
    print(f"  install to  {venv}")
    print(f"  base python {base_python} ({base_reason})")
    if base_python != sys.executable:
        print(f"              (this Python is {sys.executable}; it will not run the install)")

    if not environment.python_ok:
        print(
            "\nERROR: ClaireAI needs Python 3.11 or newer. "
            f"This is Python {environment.python}.",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    if environment.family == "other":
        print(f"\nERROR: {environment.system} is not a supported desktop OS yet.", file=sys.stderr)
        return EXIT_CONFIG
    if not environment.tkinter:
        print(f"\nWARNING: Tk is missing: {TK_HINTS[environment.family]}.")
        print("The terminal app will still work; the desktop window will not open.")
    if not environment.display and environment.family == "linux":
        print("\nWARNING: no DISPLAY or WAYLAND_DISPLAY here; use `claireai --cli` from this session.")

    requirements = requirements_for(environment)
    speech_wanted = not args.no_speech_model
    print("\nThe desktop install includes:")
    print(format_plan(requirements))
    if speech_wanted:
        print(f"  {'speech-to-text model':<24} {SPEECH_MODEL_DOWNLOAD}")
        print(f"  {'':<24} faster-whisper tiny; optional, downloaded once, stays local")
    print("\nNothing is downloaded until you answer yes. Re-run with --no-speech-model")
    print("to skip the voice model, or install it later from Settings > Voice.")

    if args.dry_run:
        print("\nDry run: no changes were made.")
        return 0
    if not ask("Download the items above and install ClaireAI?", assume_yes=args.yes):
        print("\nNothing was downloaded. Run this again when you are ready.")
        return 0

    if not python.is_file():
        print(f"\nCreating {venv} (first run only, no downloads).")
        status = stream([base_python, "-m", "venv", str(venv)])
        if status != 0 or not python.is_file():
            print("ERROR: the virtual environment could not be created.", file=sys.stderr)
            return EXIT_CONFIG

    torch = next(item for item in requirements if item.key == "torch")
    status = stream(
        install_command(str(python), torch.spec, index_url=cpu_index_url(environment))
    )
    if status != 0:
        print("ERROR: the local model runtime could not be installed.", file=sys.stderr)
        return EXIT_CONFIG

    # The complete application: window, runtime and voice input.
    status = stream(install_command(str(python), f"{ROOT / 'assistant'}[desktop]"))
    if status != 0:
        print("ERROR: ClaireAI could not be installed.", file=sys.stderr)
        return EXIT_CONFIG

    data_dir = _data_dir(python)
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        marker = data_dir / "update-source"
        marker.write_text(str((ROOT / "assistant").resolve()), encoding="utf-8")
        try:
            marker.chmod(0o600)
        except OSError:
            pass
    if speech_wanted and data_dir is not None and not speech_model_installed(data_dir):
        print(f"\nVoice input also needs its {SPEECH_MODEL_DOWNLOAD} local model.")
        if ask("Download the speech-to-text model now?", assume_yes=args.yes):
            if not install_speech_model(data_dir, python=str(python)):
                print(
                    "\nThe speech model was not downloaded. Voice input stays off; "
                    "install it later from Settings > Voice.",
                    file=sys.stderr,
                )

    if environment.family == "linux":
        # The menu entry names this venv's Python, so it does not depend on
        # PATH and cannot pick a Tk without font antialiasing.
        stream([str(python), "-m", "claireai.app", "--install-menu"])

    print("\nChecking the finished install:")
    stream([str(python), "-m", "claireai.app", "--setup"])

    launcher = "claireai.exe" if os.name == "nt" else "claireai"
    print("Installed. Start it with:")
    print(f"  {venv / ('Scripts' if os.name == 'nt' else 'bin') / launcher}")
    print("  claireai --cli     # terminal instead of the window")
    print("  claireai --install-autostart   # start with the desktop session")
    return 0


def _data_dir(python: Path) -> Path | None:
    """Ask the installed package where it keeps data, instead of guessing."""
    result = subprocess.run(
        [str(python), "-c", "from claireai.config import Settings; print(Settings().resolved_data_dir)"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
