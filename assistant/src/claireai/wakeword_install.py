"""Install the ONNX-only openWakeWord runtime without pip's tflite dependency.

openWakeWord 0.6 declares ``tflite-runtime`` unconditionally, but that package
has no Python 3.12 wheel. ClaireAI only uses the ONNX backend, so on 3.12 the
package is installed with ``--no-deps`` and the already-declared ONNX/NumPy
dependencies come from the ``wakeword`` extra.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from typing import Sequence


def installed() -> bool:
    return importlib.util.find_spec("openwakeword") is not None


def install(*, stream=print) -> int:
    if installed():
        stream("openWakeWord is already installed.")
        return 0
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "openwakeword>=0.6,<1",
    ]
    stream("Installing the ONNX-only openWakeWord package (no tflite-runtime).")
    status = subprocess.run(command, check=False).returncode
    if status != 0:
        print("openWakeWord was not installed.", file=sys.stderr)
        return status
    if not installed():
        print("openWakeWord is still not importable after installation.", file=sys.stderr)
        return 1
    try:
        from claireai.wakeword import OpenWakeWordEngine

        available = OpenWakeWordEngine().available
    except Exception as exc:  # noqa: BLE001 - report the actual install problem
        print(f"openWakeWord installed, but the adapter check failed: {exc}", file=sys.stderr)
        return 1
    if not available:
        print(
            "openWakeWord installed, but its ONNX inference class was not available.",
            file=sys.stderr,
        )
        return 1
    print(
        "openWakeWord ONNX runtime is ready. No wake-word model was downloaded; "
        "set CLAIREAI_WAKEWORD_MODEL or use the pretrained hey_jarvis model."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claireai-install-wakeword",
        description="Install openWakeWord without the unavailable tflite-runtime dependency.",
    )
    parser.parse_args(argv)
    return install()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
