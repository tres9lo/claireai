#!/usr/bin/env python3
"""Listen for a wake word for a bounded time and print detections.

    assistant/.venv/bin/python scripts/wakeword_check.py --seconds 15

This is an explicit microphone action. It does not download a model and does
not save audio. Install the optional extra and a matching openWakeWord model
first; the default ``hey_jarvis`` model is not bundled with this repository.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from claireai.config import Settings
from claireai.wakeword import OpenWakeWordEngine, WakeWordListener, WakeWordUnavailable


def main(argv=None) -> int:  # noqa: ANN001
    parser = argparse.ArgumentParser(description="Check the local wake-word engine.")
    parser.add_argument("--model", default=None, help="openWakeWord model name or path")
    parser.add_argument("--threshold", type=float, default=None, help="0 < threshold <= 1")
    parser.add_argument("--seconds", type=float, default=15.0, help="listen duration")
    args = parser.parse_args(argv)

    settings = Settings(_env_file=None)
    model = args.model or settings.wakeword_model
    threshold = settings.wakeword_threshold if args.threshold is None else args.threshold
    engine = OpenWakeWordEngine(model, threshold=threshold)
    listener = WakeWordListener(engine)
    if not listener.available:
        print(
            "Wake-word runtime is not installed. Install the optional wakeword extra "
            "and an openWakeWord model; no model was downloaded.",
            file=sys.stderr,
        )
        return 2

    stop = threading.Event()
    deadline = time.monotonic() + max(1.0, args.seconds)

    def detected(event) -> None:  # noqa: ANN001
        print(f"detected {event.phrase!r} score={event.score:.3f}", flush=True)

    print(f"listening for {model!r} for {args.seconds:.0f}s; Ctrl+C stops", flush=True)
    timer = threading.Timer(max(1.0, args.seconds), stop.set)
    timer.daemon = True
    timer.start()
    try:
        listener.run(stop=stop, on_detection=detected)
    except KeyboardInterrupt:
        stop.set()
    except WakeWordUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
