#!/usr/bin/env python3
"""Prepare the bilingual corpus, train a candidate and evaluate it.

This is the single command the owner can run later:

    assistant/.venv/bin/python brain-lab/train_claireai.py --profile full

It downloads only the bounded, licensed sources listed in ``data-sources.json``,
builds the authored bilingual corpus, merges a capped minority of external
samples, trains a fresh checkpoint into ``runtime-data/models/candidates/`` and
runs the full fixed gate: model, conversation, standard tools and extended
tools. It never installs, promotes or overwrites the active model. Installation
stays a separate, explicit owner decision after the gate passes.

Profiles differ in corpus size and model capacity, not in data rights. ``quick``
is a smoke run for this laptop; ``full`` is the practical quality run; ``huge``
is a larger-data, larger-model run that will take many hours on CPU. All three
remain bounded samples rather than bulk scrapes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "data"
CANDIDATES = REPO / "runtime-data" / "models" / "candidates"

PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "base_records": 8_000,
        "external_limit": 500,
        "external_share": 0.35,
        "steps": 1_500,
        "vocab": 2_048,
        "width": 224,
        "layers": 5,
        "heads": 7,
        "context": 512,
        "batch": 8,
        "tokenizer_chars": 2_000_000,
    },
    "full": {
        "base_records": 34_000,
        "external_limit": 4_000,
        "external_share": 0.35,
        "steps": 8_000,
        "vocab": 2_048,
        "width": 224,
        "layers": 5,
        "heads": 7,
        "context": 512,
        "batch": 8,
        "tokenizer_chars": 4_000_000,
    },
    "huge": {
        "base_records": 60_000,
        "external_limit": 8_000,
        "external_share": 0.35,
        "steps": 15_000,
        "vocab": 4_096,
        "width": 320,
        "layers": 6,
        "heads": 8,
        "context": 768,
        "batch": 8,
        "tokenizer_chars": 6_000_000,
    },
}


def _python() -> str:
    """Prefer the project environment, whatever interpreter started this script."""
    for candidate in (
        REPO / "assistant" / ".venv" / "bin" / "python",
        REPO / "assistant" / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _run(command: Sequence[str], *, cwd: Path = REPO) -> int:
    print("\n$ " + " ".join(command), flush=True)
    return subprocess.run(list(command), cwd=cwd, check=False).returncode


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _tokenizer_manifest(vocab_size: int) -> Path | None:
    """The nearest existing merges for the next corpus budget check.

    A v16 tokenizer does not exist until training starts.  Reuse a checkpoint
    sidecar with the same vocabulary so merge-time filtering and the audit use
    the same token accounting instead of a character estimate.
    """
    candidates = [
        REPO / "runtime-data" / "models" / "claire-scratch.json",
        *sorted(CANDIDATES.glob("*.json")),
    ]
    for path in candidates:
        payload = _read_manifest(path)
        architecture = payload.get("architecture")
        if (
            isinstance(architecture, dict)
            and architecture.get("vocab_size") == vocab_size
            and payload.get("tokenizer_merges")
        ):
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--name", default=None, help="candidate name (default: profile)")
    parser.add_argument("--steps", type=int, default=None, help="override the profile step count")
    parser.add_argument("--external-limit", type=int, default=None, help="records per external source")
    parser.add_argument(
        "--external-share",
        type=float,
        default=None,
        help="maximum fraction of merged training records from external sources (default: 0.35)",
    )
    parser.add_argument("--base-records", type=int, default=None, help="authored-corpus quota")
    parser.add_argument("--no-fetch", action="store_true", help="reuse files already downloaded")
    parser.add_argument("--no-build", action="store_true", help="reuse brain-lab/data/chat-v14")
    parser.add_argument("--no-merge", action="store_true", help="reuse brain-lab/data/chat-v16")
    parser.add_argument("--no-eval", action="store_true", help="train only; do not run the gate")
    parser.add_argument("--context", type=int, default=None)
    args = parser.parse_args()

    profile = dict(PROFILES[args.profile])
    if args.steps is not None:
        profile["steps"] = args.steps
    if args.external_limit is not None:
        profile["external_limit"] = args.external_limit
    if args.external_share is not None:
        profile["external_share"] = args.external_share
    if args.base_records is not None:
        profile["base_records"] = args.base_records
    if args.context is not None:
        profile["context"] = args.context
    if min(
        profile["steps"],
        profile["external_limit"],
        profile["base_records"],
        profile["context"],
    ) < 1:
        parser.error("steps, external limit, base records and context must be positive")
    if not 0.0 <= profile["external_share"] < 1.0:
        parser.error("external share must be at least 0 and below 1")

    name = args.name or f"claire-chat-v16-{args.profile}"
    base_corpus = DATA / "chat-v14"
    merged_corpus = DATA / "chat-v16"
    candidate = CANDIDATES / f"{name}.pt"
    pipeline_log = candidate.with_suffix(".pipeline.json")
    python = _python()
    started = time.time()
    status: dict[str, Any] = {
        "profile": args.profile,
        "name": name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": python,
        "steps": profile["steps"],
        "external_limit": profile["external_limit"],
        "external_share": profile["external_share"],
        "base_records": profile["base_records"],
        "context": profile["context"],
        "corpus": str(merged_corpus),
        "candidate": str(candidate),
        "stages": {},
    }

    def finish(code: int) -> int:
        status["elapsed_seconds"] = round(time.time() - started, 1)
        status["exit_code"] = code
        status["completed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        pipeline_log.parent.mkdir(parents=True, exist_ok=True)
        pipeline_log.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(f"\npipeline log: {pipeline_log}")
        return code

    if not args.no_fetch:
        code = _run(
            [
                python,
                str(ROOT / "fetch_corpora.py"),
                "--source",
                "all",
                "--limit",
                str(profile["external_limit"]),
            ]
        )
        # A rate-limited optional source is reported, not fatal: the merge below
        # still needs at least one licensed external file.
        status["stages"]["fetch"] = code
        if code not in (0, 2):
            return finish(code)

    if not args.no_build:
        code = _run(
            [
                python,
                str(ROOT / "build_chat_dataset.py"),
                "--output",
                str(base_corpus),
                "--total-records",
                str(profile["base_records"]),
            ]
        )
        status["stages"]["build"] = code
        if code != 0:
            return finish(code)

    if not args.no_merge:
        merge_command = [
            python,
            str(ROOT / "merge_corpora.py"),
            "--base",
            str(base_corpus),
            "--external",
            str(DATA / "external"),
            "--output",
            str(merged_corpus),
            "--limit",
            str(profile["external_limit"]),
            "--external-share",
            str(profile["external_share"]),
            "--context",
            str(profile["context"]),
            "--target-tokens",
            "192",
        ]
        tokenizer_manifest = _tokenizer_manifest(profile["vocab"])
        if tokenizer_manifest is not None:
            merge_command += ["--tokenizer-manifest", str(tokenizer_manifest)]
        code = _run(merge_command)
        status["stages"]["merge"] = code
        status["merge_command"] = merge_command
        if code != 0:
            return finish(code)

    if not (merged_corpus / "train.jsonl").is_file():
        print(f"ERROR: {merged_corpus} has no train.jsonl; run without --no-merge", file=sys.stderr)
        return finish(2)
    status["corpus_manifest"] = _read_manifest(merged_corpus / "manifest.json")
    status["records_by_split"] = status["corpus_manifest"].get("records_by_split", {})

    train_command = [
        python,
        str(ROOT / "train_chat.py"),
        "--corpus",
        str(merged_corpus),
        "--steps",
        str(profile["steps"]),
        "--batch-size",
        str(profile["batch"]),
        "--context",
        str(profile["context"]),
        "--vocab-size",
        str(profile["vocab"]),
        "--width",
        str(profile["width"]),
        "--layers",
        str(profile["layers"]),
        "--heads",
        str(profile["heads"]),
        "--max-target-tokens",
        "192",
        "--tokenizer-chars",
        str(profile["tokenizer_chars"]),
        "--output",
        str(candidate),
        "--name",
        name,
    ]
    code = _run(train_command)
    status["stages"]["train"] = code
    status["train_command"] = train_command
    if code != 0 or not candidate.is_file():
        return finish(code or 2)

    if not args.no_eval:
        gate_dir = REPO / "runtime-data" / "eval" / f"{name}-gate"
        gate_command = [
            python,
            str(REPO / "scripts" / "run_gate.py"),
            "--checkpoint",
            str(candidate),
            "--out-dir",
            str(gate_dir),
            "--runs",
            "3",
            "--conversation-runs",
            "3",
        ]
        gate_code = _run(gate_command)
        status["stages"]["gate"] = gate_code
        status["gate_command"] = gate_command
        status["gate_report"] = str(gate_dir / "gate.json")
        if gate_code != 0:
            print(
                "\nCandidate failed the full post-training gate. It was left in "
                f"{candidate} and the active model was not changed.",
                file=sys.stderr,
            )
            return finish(1)

    if args.no_eval:
        print(
            "\nCandidate trained; gate was skipped. It is NOT installed. "
            "The active checkpoint and permissions were not changed."
        )
    else:
        print(
            "\nCandidate trained and passed the full gate. It is NOT installed. "
            "The active checkpoint and permissions were not changed."
        )
    return finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
