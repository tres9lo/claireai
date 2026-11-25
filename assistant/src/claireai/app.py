"""One command to run ClaireAI.

    claireai                # open the native desktop window
    claireai --cli          # the terminal chat
    claireai --setup        # report OS, Python, packages and the model
    claireai --setup --install   # offer the missing packages, with sizes
    claireai --once "..."   # any existing CLI flag still works

The default surface is the native Tk window and the terminal. A loopback HTTP
API is available only with ``--api``; it requires a bearer token. Nothing here
is privileged or downloads anything on its own.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from typing import Callable, Sequence

from claireai.config import Settings
from claireai.preflight import (
    SPEECH_MODEL_DOWNLOAD,
    ask_to_install,
    detect_environment,
    format_environment,
    format_plan,
    install,
    missing_parts,
    missing_requirements,
    requirements_for,
    speech_model_installed,
)
from claireai.storage import Database

EXIT_OK = 0
EXIT_CONFIG = 2

# Flags the removed browser surface used to own. Saying so is kinder than
# passing them to the CLI, where they would fail as unknown model flags.
REMOVED_FLAGS = {"--port", "--no-browser", "--ui"}


def _distribution_version(spec: str) -> str:
    if not spec.strip():
        return "system install"
    name = spec.split(">=", 1)[0].split("<", 1)[0].strip()
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def setup_report(
    settings: Settings,
    *,
    voice_report: Callable[[], str] | None = None,
    install_missing: bool = False,
    assume_yes: bool = False,
) -> int:
    """Detect the host, report readiness, and install only after consent.

    ``--setup`` on its own never downloads. ``--setup --install`` prints every
    component and its approximate size, then asks; ``--yes`` is the explicit
    non-interactive form of that consent.
    """
    settings.ensure_directories()
    applied = Database(settings.database_path).migrate()
    environment = detect_environment()
    requirements = requirements_for(environment)
    missing = missing_parts(
        settings.resolved_data_dir,
        models=(settings.brain_model, settings.brain_coding_model),
        embedding_model=settings.embedding_model if settings.memory_enabled else None,
    )

    if voice_report is None:

        def voice_report() -> str:
            from claireai.voice import available_voices

            voices = available_voices()
            return (
                ", ".join(sorted(voices.values()))
                if voices
                else "off (install eSpeak NG or add a Piper voice)"
            )

    speech = (
        "installed"
        if speech_model_installed(settings.resolved_data_dir)
        else f"not downloaded ({SPEECH_MODEL_DOWNLOAD}; installed by --setup --install or Settings > Voice)"
    )

    lines = [
        "",
        "ClaireAI setup",
        "  host",
        format_environment(environment),
        "",
        f"  data        {settings.resolved_data_dir}",
        f"  database    {settings.database_path}",
        f"  migrations  {', '.join(applied) if applied else 'already up to date'}",
        f"  brain       {settings.brain_provider} @ {settings.resolved_brain_base_url or 'local checkpoint'}",
        f"  model       {settings.brain_model or settings.resolved_model_checkpoint}",
        f"  coding      {settings.brain_coding_model or '(same as main model)'}",
        f"  speech      {speech}",
        f"  voice       {voice_report()}",
        f"  memory      {'on · ' + settings.embedding_model if settings.memory_enabled else 'off'}",
        f"  API         on demand at http://{settings.api_host}:{settings.api_port}",
        "",
        "  packages",
    ]
    lines.extend(
        f"              {item.label:<24} {_distribution_version(item.spec)}"
        for item in requirements
    )
    lines.append("")

    if not missing:
        lines.append("Ready. Run `claireai` for the window, or `claireai --cli` for the terminal.")
    else:
        lines.append(
            "The model runtime is not complete yet; the missing parts are listed below. "
            "Nothing is downloaded until you approve it."
        )

    print("\n".join(lines))

    if not missing:
        if not environment.desktop_ready:
            print("\nNo desktop display was found; `claireai --cli` works here.")
        return EXIT_OK

    core = {item.key for item in missing} & {"ollama", "ollama_models", "customtkinter"}
    if install_missing:
        consent = assume_yes
        if not consent:
            consent = ask_to_install(missing)
        if consent:
            print("\nDownloading now. Each package and its real size is printed as it arrives:")
            failed = install(missing, data_dir=settings.resolved_data_dir)
            still_missing = {item.key for item in failed}
            if still_missing:
                print(
                    "\nNot installed: "
                    + ", ".join(sorted(still_missing))
                    + ". The app can still start only where those parts are not needed; "
                    "run this again when the network is available.",
                    file=sys.stderr,
                )
            else:
                print("\nAll requested parts are installed.")
            return EXIT_OK if not (still_missing & core) else EXIT_CONFIG
        print("\nNothing was downloaded. Install later with `claireai --setup --install`.")
        return EXIT_OK if not core else EXIT_CONFIG

    print("Missing parts (nothing is downloaded until you say yes):")
    print(format_plan(missing))
    print("Install them with `claireai --setup --install` (add `--yes` to skip the prompt).")
    return EXIT_OK if not core else EXIT_CONFIG


# Flags this launcher owns. Anything else is a CLI flag and is passed through
# untouched, which keeps `claireai --once "..."` working.
APP_FLAGS = frozenset(
    {
        "--cli", "-i", "--data-dir", "-h", "--help",
        "--desktop", "--install-autostart", "--uninstall-autostart", "--autostart-status",
        "--install-menu", "--uninstall-menu",
        "--setup", "--install", "--yes", "--update", "--source",
        "--training-run", "--install-training-service", "--uninstall-training-service",
        "--training-service-status",
        "--api", "--api-host", "--api-port",
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    """Open the window, or hand the arguments to the CLI unchanged."""
    args = list(sys.argv[1:] if argv is None else argv)
    first = args[0].split("=", 1)[0] if args else ""

    if first in {"--cli", "-i"}:  # the terminal chat
        from claireai.cli import main as cli_main

        return cli_main([])
    if first in REMOVED_FLAGS:
        print(
            f"ERROR: {first} belonged to the browser UI, which has been removed. "
            "Use `claireai` for the desktop window or `claireai --cli` for the terminal.",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    if args and first not in APP_FLAGS:
        from claireai.cli import main as cli_main

        return cli_main(args)

    import argparse

    parser = argparse.ArgumentParser(
        prog="claireai", description="Open the native ClaireAI app (desktop or terminal)."
    )
    parser.add_argument("--cli", "-i", action="store_true", help="open the terminal chat instead of the window")
    parser.add_argument("--desktop", action="store_true", help="open the native window (the default)")
    parser.add_argument("--data-dir", help="data directory (default: the OS application-data directory)")
    parser.add_argument("--install-autostart", action="store_true", help="start ClaireAI with the desktop session")
    parser.add_argument("--uninstall-autostart", action="store_true", help="stop starting it with the session")
    parser.add_argument("--autostart-status", action="store_true", help="report the autostart entry")
    parser.add_argument("--install-menu", action="store_true", help="add ClaireAI to the applications menu")
    parser.add_argument("--uninstall-menu", action="store_true", help="remove the applications menu entry")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="detect the OS, Python, packages, model and voice, then exit",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="with --setup: list missing parts and download them after a yes",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="with --setup --install: treat the download list as approved",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rebuild this checkout and reinstall it over the running copy, then restart",
    )
    parser.add_argument(
        "--source",
        help="with --update: the checkout root or its assistant directory (remembered afterwards)",
    )
    parser.add_argument(
        "--training-run",
        action="store_true",
        help="run one headless training attempt (used by the user timer)",
    )
    parser.add_argument(
        "--install-training-service",
        action="store_true",
        help="install a daily systemd user timer for training",
    )
    parser.add_argument(
        "--uninstall-training-service",
        action="store_true",
        help="remove the daily training timer",
    )
    parser.add_argument(
        "--training-service-status",
        action="store_true",
        help="report the training timer state",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="run the loopback HTTP API instead of opening a window",
    )
    parser.add_argument("--api-host", help="API bind address (default 127.0.0.1)")
    parser.add_argument("--api-port", type=int, help="API port (default 8765)")
    parsed = parser.parse_args(args)
    if parsed.install and not parsed.setup:
        parser.error("--install is only valid together with --setup")
    if parsed.source and not parsed.update:
        parser.error("--source is only valid together with --update")

    if parsed.update:
        from claireai.update import UpdateError, run_update

        try:
            result = run_update(
                source=Path(parsed.source).expanduser() if parsed.source else None
            )
        except UpdateError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        print(f"Updated from {result['source']} ({result['wheel']}).")
        if result.get("warning"):
            print(
                "WARNING: the code was replaced but a dependency could not be "
                f"installed: {result['warning']}",
                file=sys.stderr,
            )
        print("Restart ClaireAI to run the new code.")
        return EXIT_OK

    if parsed.setup:
        settings = (
            Settings(data_dir=Path(parsed.data_dir).expanduser())
            if parsed.data_dir
            else Settings()
        )
        return setup_report(
            settings,
            install_missing=parsed.install,
            assume_yes=parsed.yes,
        )

    if parsed.api:
        from claireai.api.main import run_api

        return run_api(
            host=parsed.api_host,
            port=parsed.api_port,
            data_dir=Path(parsed.data_dir).expanduser() if parsed.data_dir else None,
        )

    if (
        parsed.training_run
        or parsed.install_training_service
        or parsed.uninstall_training_service
        or parsed.training_service_status
    ):
        settings = (
            Settings(data_dir=Path(parsed.data_dir).expanduser())
            if parsed.data_dir
            else Settings()
        )
        from claireai import service

        if parsed.training_run:
            from claireai.training import run_service_tick

            result = run_service_tick(settings)
            print(
                f"training: {result.get('status', 'not started')} "
                f"(enabled={result.get('enabled')}, ran={result.get('ran')})"
            )
            if result.get("candidate"):
                print(f"candidate: {result['candidate']}")
            gate = result.get("gate") or {}
            if gate.get("reason"):
                print(f"gate: {gate['reason']}")
            if result.get("worker_error"):
                print(f"training problem: {result['worker_error']}", file=sys.stderr)
                return EXIT_CONFIG
            return EXIT_OK
        if parsed.training_service_status:
            state = service.status()
            print(f"training service {'installed' if state['installed'] else 'not installed'}")
            for key in ("active", "enabled", "next"):
                print(f"  {key}: {state.get(key)}")
            print(f"  timer: {state['timer']}")
            return EXIT_OK
        if parsed.uninstall_training_service:
            removed = service.uninstall()
            print("training service removed" if removed else "no training service was installed")
            return EXIT_OK
        try:
            from claireai.update import checkout_root

            workdir = checkout_root(settings.resolved_data_dir)
            paths = service.install(
                settings.resolved_data_dir,
                working_directory=workdir or Path.cwd(),
            )
        except service.ServiceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        print(f"training timer installed: {paths['timer']}")
        print("It runs at most once a day when Background learning is on.")
        print("Check it with: claireai --training-service-status")
        return EXIT_OK

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
    if parsed.install_menu:
        from claireai import menu

        target = menu.install()
        print(f"ClaireAI was added to the applications menu: {target}")
        print(f"It runs: {menu.status()['command']}")
        return EXIT_OK
    if parsed.uninstall_menu:
        from claireai import menu

        removed = menu.uninstall()
        print("menu entry removed" if removed else "no menu entry was installed")
        return EXIT_OK

    settings = (
        Settings(data_dir=Path(parsed.data_dir).expanduser())
        if parsed.data_dir
        else Settings()
    )
    # Only the legacy scratch provider needs a checkpoint copied in. The normal
    # Ollama path owns its model files under Ollama's own directory.
    if settings.brain_provider == "scratch":
        try:
            from claireai.models import ensure_installed_checkpoint

            ensure_installed_checkpoint(settings)
        except Exception:  # noqa: BLE001 - the window explains a missing checkpoint
            pass

    # The window is useless without its model runtime. Say exactly what is
    # missing and how to get it instead of opening a chat that cannot answer.
    missing = missing_requirements()
    runtime = next((item for item in missing if item.key == "ollama"), None)
    if runtime is None and settings.brain_provider == "ollama":
        from claireai.preflight import ollama_models_installed

        wanted = (settings.brain_model, settings.brain_coding_model)
        if not ollama_models_installed(wanted):
            print("ClaireAI needs its Qwen models before the window can answer:", file=sys.stderr)
            print(
                f"  ollama pull {settings.brain_model}\n"
                f"  ollama pull {settings.brain_coding_model}",
                file=sys.stderr,
            )
            print("Then run `claireai --setup` to check the install.", file=sys.stderr)
            return EXIT_CONFIG
    if runtime is not None:
        print("ClaireAI needs Ollama before the window can answer:", file=sys.stderr)
        print(format_plan([runtime]), file=sys.stderr)
        print("Install it with `claireai --setup --install`, or `./install.sh` to set up everything.", file=sys.stderr)
        return EXIT_CONFIG

    # The launcher, not the window module: it reports a missing Tk toolkit
    # instead of raising ModuleNotFoundError while importing it. The Tk surface
    # remains the desktop UI; the API is a separate opt-in service.
    from claireai.desktop.launcher import main as desktop_main

    return desktop_main(["--data-dir", str(settings.resolved_data_dir)])


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
