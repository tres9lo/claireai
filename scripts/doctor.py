#!/usr/bin/env python3
"""Read-only prerequisite and configuration diagnostics.

    assistant/.venv/bin/python scripts/doctor.py
    assistant/.venv/bin/python scripts/doctor.py --json

Prints versions, paths, database/schema state and local model-runtime status.
It never creates a database, downloads a model, writes config, or prints secret
values (only the *names* of set CLAIREAI_* variables).

Exit codes: 0 ready, 1 something missing or not ready, 2 invalid configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from claireai.doctor import collect, format_text, report_as_dict


async def _run(as_json: bool) -> int:
    report = await collect()
    if as_json:
        print(json.dumps(report_as_dict(report), indent=2))
    else:
        print(format_text(report))
    return report.exit_code


def main(argv=None) -> int:  # noqa: ANN001
    parser = argparse.ArgumentParser(prog="doctor", description="Read-only ClaireAI diagnostics.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args.json))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
