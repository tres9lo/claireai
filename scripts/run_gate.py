#!/usr/bin/env python3
"""Run the fixed post-training gate and save every report in one place.

This does not install a checkpoint. It only measures:

1. one-shot model cases;
2. multi-turn conversation cases;
3. the standard tool-call suite;
4. the extended direct-answer / missing-argument / refusal probes.

    assistant/.venv/bin/python scripts/run_gate.py \
        --checkpoint runtime-data/models/candidates/claire-chat-v16-full.pt

Exit codes: 0 all stages passed, 1 at least one stage failed, 2 bad configuration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    report: Path


def build_stages(
    checkpoint: Path,
    output: Path,
    *,
    python: str = sys.executable,
    runs: int = 3,
    conversation_runs: int = 1,
    include_model: bool = True,
    include_conversation: bool = True,
    include_tools: bool = True,
) -> list[Stage]:
    stages: list[Stage] = []
    if include_model:
        stages.append(
            Stage(
                "model",
                [
                    python,
                    str(SCRIPTS / "eval_model.py"),
                    "--model-checkpoint",
                    str(checkpoint),
                    "--runs",
                    str(runs),
                    "--out",
                    str(output / "model.json"),
                ],
                output / "model.json",
            )
        )
    if include_conversation:
        stages.append(
            Stage(
                "conversation",
                [
                    python,
                    str(SCRIPTS / "conversation_eval.py"),
                    "--model-checkpoint",
                    str(checkpoint),
                    "--runs",
                    str(conversation_runs),
                    "--out",
                    str(output / "conversation.json"),
                ],
                output / "conversation.json",
            )
        )
    if include_tools:
        stages.extend(
            [
                Stage(
                    "tool-calls",
                    [
                        python,
                        str(SCRIPTS / "tool_call_eval.py"),
                        "--model-checkpoint",
                        str(checkpoint),
                        "--runs",
                        str(runs),
                        "--cases",
                        str(SCRIPTS / "tool_call_cases.json"),
                        "--out",
                        str(output / "tool-calls.json"),
                    ],
                    output / "tool-calls.json",
                ),
                Stage(
                    "tool-calls-extended",
                    [
                        python,
                        str(SCRIPTS / "tool_call_eval.py"),
                        "--model-checkpoint",
                        str(checkpoint),
                        "--runs",
                        str(runs),
                        "--cases",
                        str(SCRIPTS / "tool_call_cases_extended.json"),
                        "--out",
                        str(output / "tool-calls-extended.json"),
                    ],
                    output / "tool-calls-extended.json",
                ),
            ]
        )
    return stages


def run_stages(
    stages: Sequence[Stage],
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    fail_fast: bool = False,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for stage in stages:
        completed = runner(stage.command, cwd=ROOT, check=False)
        result = {
            "name": stage.name,
            "command": stage.command,
            "report": str(stage.report),
            "exit_code": int(completed.returncode),
            "passed": completed.returncode == 0,
        }
        results.append(result)
        if fail_fast and completed.returncode != 0:
            break
    return results


def _result_payload(
    checkpoint: Path, output: Path, results: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": bool(results) and all(bool(item["passed"]) for item in results),
        "stages": results,
        "install_rule": "Install only when every stage above passed.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--conversation-runs", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-conversation", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.conversation_runs < 1:
        parser.error("--runs and --conversation-runs must be positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    output = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else ROOT / "runtime-data" / "eval" / f"gate-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    stages = build_stages(
        checkpoint,
        output,
        python=args.python,
        runs=args.runs,
        conversation_runs=args.conversation_runs,
        include_model=not args.skip_model,
        include_conversation=not args.skip_conversation,
        include_tools=not args.skip_tools,
    )
    if not stages:
        parser.error("every gate stage was skipped")
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint is not a file: {checkpoint}", file=sys.stderr)
        return EXIT_CONFIG
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "stages": [
                        {"name": stage.name, "command": stage.command, "report": str(stage.report)}
                        for stage in stages
                    ],
                },
                indent=2,
            )
        )
        return EXIT_OK

    for stage in stages:
        print(f"=== {stage.name} ===")
        print(" ".join(stage.command))
    results = run_stages(stages, fail_fast=args.fail_fast)
    payload = _result_payload(checkpoint, output, results)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"gate report: {output / 'gate.json'}")
    return EXIT_OK if payload["passed"] else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
