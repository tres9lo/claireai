"""What this machine needs before the app can work, and what it would download.

The rule this module exists to enforce: **nothing is downloaded without the
owner seeing the list first and answering yes**. Installing an app must not be
a surprise 700 MB download, and a missing optional runtime must be reported in
plain language instead of surfacing as a traceback halfway through a turn.

Nothing here imports torch, customtkinter or faster-whisper, so it stays
importable on a bare Python and can be reused by the bootstrap installer before
the package is installed. Downloads happen only inside :func:`install`, which is
called after :func:`ask_to_install` has printed exactly what will be fetched.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

MINIMUM_PYTHON = (3, 11)

# Sizes are deliberately approximate: wheel sizes change between releases, and
# a range that is close is more honest than a single number that will be wrong.
# The streamed pip output shows the real size of each file as it downloads.
@dataclass(frozen=True)
class Requirement:
    key: str
    label: str
    module: str
    spec: str
    download: str
    detail: str = ""
    index_url: str = ""
    # A system executable (Ollama) rather than an importable Python module.
    command: str = ""
    # Models that `ollama pull` must fetch for this requirement.
    models: tuple[str, ...] = ()


OLLAMA_INSTALL_URL = "https://ollama.com/download"
OLLAMA_MODELS_DOWNLOAD = "about 1.5 GB for both Qwen models"
DEFAULT_OLLAMA_MODELS = ("qwen3:1.7b", "qwen2.5-coder:1.5b")

REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        key="ollama",
        label="Ollama runtime",
        module="",
        spec="",
        command="ollama",
        download="manual install, about 1 GB on disk",
        detail=f"serves Qwen locally; install from {OLLAMA_INSTALL_URL}",
    ),
    Requirement(
        key="customtkinter",
        label="desktop window",
        module="customtkinter",
        spec="customtkinter>=5.2,<7",
        download="about 2 MB",
        detail="draws the native window",
    ),
    Requirement(
        key="faster_whisper",
        label="voice input",
        module="faster_whisper",
        spec="faster-whisper>=1.1,<2",
        download="about 90 MB",
        detail="push-to-talk transcription; the speech model is a separate opt-in",
    ),
)

SPEECH_MODEL_DOWNLOAD = "about 75 MB"
SPEECH_MODEL_KEY = "speech_model"
EMBEDDING_MODEL_DOWNLOAD = "about 275 MB"


@dataclass(frozen=True)
class Environment:
    family: str
    system: str
    release: str
    machine: str
    python: str
    python_ok: bool
    display: str
    tkinter: bool
    audio: str
    tk_fonts: str = "unknown"

    @property
    def desktop_ready(self) -> bool:
        return self.tkinter and (self.family != "linux" or bool(self.display))


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # a half-removed package can raise here
        return False


def tk_font_backend() -> str:
    """``xft``, ``x11`` (bitmap-only) or why the check could not run.

    A conda Tk built without Xft reports ``x11`` and renders every glyph from a
    bitmap face, which is the pixelated-window problem. The setup report names
    it before the owner launches the window.
    """
    if platform.system().lower() != "linux":
        return "native"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no-display"
    try:
        import tkinter

        root = tkinter.Tk()
    except Exception:  # noqa: BLE001 - a broken display is reported elsewhere
        return "unavailable"
    try:
        root.withdraw()
        return str(root.tk.call("::tk::pkgconfig", "get", "fontsystem") or "unknown")
    except Exception:  # noqa: BLE001 - Tk without pkgconfig is not the conda case
        return "unknown"
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def detect_environment() -> Environment:
    """OS, Python, display and audio facts; no package imports beyond the stdlib."""
    family = {
        "windows": "windows",
        "linux": "linux",
        "darwin": "macos",
    }.get(platform.system().lower(), "other")
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or ""
    voices = [tool for tool in ("espeak-ng", "espeak", "paplay", "piper") if shutil.which(tool)]
    return Environment(
        family=family,
        system=platform.system() or "Unknown",
        release=platform.release(),
        machine=platform.machine(),
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        python_ok=sys.version_info[:2] >= MINIMUM_PYTHON,
        display=display,
        tkinter=_module_present("tkinter"),
        audio=", ".join(voices) if voices else "none found (speech output is off)",
        tk_fonts=tk_font_backend() if _module_present("tkinter") else "missing",
    )


def _requirement_present(item: Requirement) -> bool:
    if item.command and shutil.which(item.command):
        return True
    return bool(item.module) and _module_present(item.module)


def missing_requirements() -> list[Requirement]:
    return [item for item in REQUIREMENTS if not _requirement_present(item)]


def ollama_models_installed(models: Sequence[str]) -> bool:
    """Whether `ollama list` already has every configured Qwen tag."""
    wanted = tuple(str(model).strip() for model in models if str(model).strip())
    if not wanted:
        return True
    binary = shutil.which("ollama")
    if not binary:
        return False
    try:
        completed = subprocess.run(
            [binary, "list"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if completed.returncode != 0:
        return False
    available = {
        line.split()[0]
        for line in (completed.stdout or "").splitlines()
        if line.strip() and not line.lower().startswith("name")
    }
    return all(model in available or f"{model}:latest" in available for model in wanted)


def ollama_model_requirement(
    models: Sequence[str],
    *,
    label: str = "Qwen models",
    download: str = OLLAMA_MODELS_DOWNLOAD,
) -> Requirement:
    wanted = tuple(str(model).strip() for model in models if str(model).strip())
    return Requirement(
        key="ollama_models",
        label=label,
        module="",
        spec="",
        download=download,
        detail="ollama pull " + " && ollama pull ".join(wanted),
        models=wanted,
    )


def speech_requirement() -> Requirement:
    """The opt-in speech-to-text model, downloaded from its public repository."""
    return Requirement(
        key=SPEECH_MODEL_KEY,
        label="speech-to-text model",
        module="",
        spec="",
        download=SPEECH_MODEL_DOWNLOAD,
        detail="faster-whisper tiny, for push-to-talk; stays on this machine",
    )


def missing_parts(
    data_dir: Path | None = None,
    models: Sequence[str] | None = None,
    embedding_model: str | None = None,
) -> list[Requirement]:
    """Everything a full install needs, including model weights on disk."""
    parts = missing_requirements()
    wanted = tuple(models or DEFAULT_OLLAMA_MODELS)
    if not ollama_models_installed(wanted):
        parts.append(ollama_model_requirement(wanted))
    embedding = str(embedding_model or "").strip()
    if embedding and not ollama_models_installed((embedding,)):
        parts.append(
            ollama_model_requirement(
                (embedding,),
                label="embedding model",
                download=EMBEDDING_MODEL_DOWNLOAD,
            )
        )
    if data_dir is not None and not speech_model_installed(data_dir):
        parts.append(speech_requirement())
    return parts


def format_environment(environment: Environment) -> str:
    display = environment.display or "none (terminal mode works)"
    python = environment.python + ("" if environment.python_ok else " (needs 3.11+)")
    if not environment.tkinter:
        tk = "missing"
    elif environment.tk_fonts == "x11":
        tk = "found, but bitmap fonts only — run ./install.sh for antialiased text"
    elif environment.tk_fonts == "xft":
        tk = "found (xft, antialiased)"
    else:
        tk = f"found ({environment.tk_fonts})"
    return "\n".join(
        [
            f"  system      {environment.system} {environment.release} ({environment.machine})",
            f"  python      {python}",
            f"  display     {display}",
            f"  tk          {tk}",
            f"  audio       {environment.audio}",
        ]
    )


def format_plan(requirements: Sequence[Requirement], *, indent: str = "  ") -> str:
    """One line per component, with its download size and why it is needed."""
    lines = []
    for item in requirements:
        lines.append(f"{indent}{item.label:<24} {item.download}")
        if item.detail:
            lines.append(f"{indent}{'':<24} {item.detail}")
    return "\n".join(lines)


def ask_to_install(
    requirements: Sequence[Requirement],
    *,
    stream: Callable[[str], None] = print,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Show the list, then ask. The default is no, and a non-tty never blocks."""
    if not requirements:
        return False
    stream("")
    stream("These parts are missing. Downloading them needs the network:")
    stream(format_plan(requirements))
    stream("")
    if not sys.stdin.isatty():
        stream("Nothing was downloaded. Run `claireai --setup --install` at a terminal to continue.")
        return False
    stream("Nothing is downloaded until you answer yes. You can install later with")
    stream("`claireai --setup --install`, or skip and use what already works.")
    answer = input_fn("Download and install these now? [y/N]: ").strip().lower()
    accepted = answer in {"y", "yes"}
    if not accepted:
        stream("Nothing was downloaded. Install later with `claireai --setup --install`.")
    return accepted


def install(
    requirements: Sequence[Requirement],
    *,
    data_dir: Path | None = None,
    python: str | None = None,
    stream: Callable[[str], None] = print,
    runner: Callable[[Sequence[str]], int] | None = None,
) -> list[Requirement]:
    """Install each requirement in turn, streaming the real download lines."""
    failed: list[Requirement] = []
    for item in requirements:
        if item.key == "ollama":
            # Ollama is a system runtime, not a Python wheel. Do not run a
            # platform-specific installer behind a generic consent prompt.
            stream("")
            stream(f"Ollama needs a manual install from {OLLAMA_INSTALL_URL}")
            stream("Install it for this OS, then run `claireai --setup --install` again.")
            failed.append(item)
            continue
        if item.key == "ollama_models":
            binary = shutil.which("ollama")
            if not binary:
                stream("  Ollama is not on PATH; install it before pulling models.")
                failed.append(item)
                continue
            ok = True
            for model in item.models:
                stream("")
                stream(f"Pulling {model} ({item.download})")
                command = [binary, "pull", model]
                try:
                    status = runner(command) if runner else _stream_command(command, stream)
                except OSError as exc:
                    stream(f"  could not start ollama: {exc}")
                    ok = False
                    continue
                if status != 0:
                    stream(f"  {model} was not pulled (ollama exited {status}).")
                    ok = False
            if not ok:
                failed.append(item)
            continue
        if item.key == SPEECH_MODEL_KEY:
            if data_dir is None:
                stream(f"  {item.label} needs a data directory; skipped.")
                failed.append(item)
            elif not install_speech_model(
                data_dir, python=python, stream=stream, runner=runner
            ):
                failed.append(item)
            continue

        command = [python or sys.executable, "-m", "pip", "install", item.spec]
        if item.index_url:
            command[4:4] = ["--index-url", item.index_url]
        stream("")
        stream(f"Installing {item.label}: {item.spec} ({item.download})")
        try:
            status = runner(command) if runner else _stream_command(command, stream)
        except OSError as exc:
            stream(f"  could not start pip: {exc}")
            failed.append(item)
            continue
        if status != 0:
            stream(f"  {item.label} was not installed (pip exited {status}).")
            failed.append(item)
    return failed


def install_speech_model(
    data_dir: Path,
    *,
    python: str | None = None,
    stream: Callable[[str], None] = print,
    runner: Callable[[Sequence[str]], int] | None = None,
) -> bool:
    """Download the tiny speech model into the data directory, showing progress."""
    cache = Path(data_dir) / "models" / "voice" / "asr"
    script = (
        "import sys;"
        "from claireai.voice_input import transcriber_for;"
        "transcriber_for(sys.argv[1]).install();"
        "print('speech model ready')"
    )
    command = [python or sys.executable, "-c", script, str(cache)]
    stream("")
    stream(f"Downloading the {SPEECH_MODEL_DOWNLOAD} speech-to-text model into {cache}")
    try:
        status = runner(command) if runner else _stream_command(command, stream)
    except OSError as exc:
        stream(f"  could not start the speech model download: {exc}")
        return False
    if status != 0:
        stream(f"  the speech model was not downloaded (exit {status}).")
        return False
    return True


def _stream_command(command: Sequence[str], stream: Callable[[str], None]) -> int:
    """Run pip with output shown as it arrives, so names and sizes are visible."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        stream(line.rstrip("\n"))
    return process.wait()


def speech_model_installed(data_dir: Path) -> bool:
    """Whether the opt-in faster-whisper model is already on disk."""
    root = Path(data_dir) / "models" / "voice" / "asr" / "models--Systran--faster-whisper-tiny" / "snapshots"
    return any((snapshot / "model.bin").is_file() for snapshot in root.glob("*"))


def cpu_index_url(environment: Environment) -> str:
    """The PyPI default on Linux pulls CUDA wheels; the CPU index is much smaller."""
    if environment.family in {"linux", "windows"} and environment.machine.lower() in {
        "x86_64",
        "amd64",
    }:
        return "https://download.pytorch.org/whl/cpu"
    return ""


def requirements_for(environment: Environment | None = None) -> list[Requirement]:
    """Requirements with the right torch source for this host."""
    environment = environment or detect_environment()
    index = cpu_index_url(environment)
    return [
        replace(item, index_url=index) if item.key == "torch" and index else item
        for item in REQUIREMENTS
    ]


__all__ = [
    "DEFAULT_OLLAMA_MODELS",
    "EMBEDDING_MODEL_DOWNLOAD",
    "Environment",
    "MINIMUM_PYTHON",
    "OLLAMA_INSTALL_URL",
    "OLLAMA_MODELS_DOWNLOAD",
    "REQUIREMENTS",
    "Requirement",
    "SPEECH_MODEL_DOWNLOAD",
    "ask_to_install",
    "cpu_index_url",
    "detect_environment",
    "format_environment",
    "format_plan",
    "install",
    "SPEECH_MODEL_KEY",
    "missing_parts",
    "missing_requirements",
    "ollama_model_requirement",
    "ollama_models_installed",
    "requirements_for",
    "speech_model_installed",
    "speech_requirement",
    "tk_font_backend",
    "install_speech_model",
]
