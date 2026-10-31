#!/usr/bin/env python3
"""Measure what the configured local model can actually do.

This is a read-only diagnostic: it sends the fixed cases in
``scripts/model_eval_cases.json`` to the local brain and scores every reply with
deterministic checks. Nothing it reports is a product claim on its own — record
the results in PROGRESS.md and use them to decide which features may rely on the
model and which must be plain code.

    assistant/.venv/bin/python scripts/eval_model.py --stub        # wiring check
    assistant/.venv/bin/python scripts/eval_model.py --runs 3      # real model

Exit codes: 0 all cases stable, 1 some case failed/flaked, 2 bad configuration,
3 model runtime unavailable or model not installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from claireai.brain import Brain, BrainHealth, BrainResponse, OllamaBrain
from claireai.config import Settings
from claireai.evaluation import (
    load_cases,
    render_expected_reply,
    render_report,
    run_cases,
    summarise,
)

DEFAULT_CASES = Path(__file__).resolve().parent / "model_eval_cases.json"

EXIT_OK = 0
EXIT_CASE_FAILURES = 1
EXIT_CONFIG = 2
EXIT_RUNTIME = 3


class StubBrain:
    """Answers each case with a reply that satisfies it by construction.

    Proves the harness plumbing (loading, running, scoring, reporting, saving).
    It says nothing about model quality and is labelled as such in the report.
    """

    name = "stub"

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    async def health(self) -> BrainHealth:
        return BrainHealth(available=True, model_present=True, models=("stub",), detail="stub")

    async def generate(self, messages, tools=None, *, temperature=None) -> BrainResponse:  # noqa: ANN001
        for message in reversed(messages):
            if message.role == "user":
                return BrainResponse(content=self._answers.get(message.content, "STUB"), model="stub")
        return BrainResponse(content="STUB", model="stub")


def _build_brain(args: argparse.Namespace, cases) -> tuple[Brain, str, Settings | None]:
    if args.stub:
        answers = {case.prompt: render_expected_reply(case) for case in cases}
        label = "STUB MODE - harness wiring check, NOT a model result"
        return StubBrain(answers), label, None

    overrides: dict[str, object] = {"ollama_timeout_seconds": 300.0}
    if args.model:
        overrides["ollama_model"] = args.model
    if args.base_url:
        overrides["ollama_base_url"] = args.base_url
    settings = Settings(**overrides)
    brain = OllamaBrain(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
        # Deterministic decoding: this harness measures capability, not luck.
        options={"temperature": 0, "num_predict": 200},
    )
    return brain, f"model {settings.ollama_model} @ {settings.ollama_base_url}", settings


def _save(results, *, path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "summary": summarise(results),
        "cases": [
            {
                "id": result.case_id,
                "category": result.category,
                "passed": result.passed,
                "runs": result.runs,
                "failures": list(result.failures),
                "sample": result.sample,
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


async def _run(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(Path(args.cases))
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load cases: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    brain, label, settings = _build_brain(args, cases)

    try:
        if not args.stub:
            health = await brain.health()  # type: ignore[attr-defined]
            if not health.available:
                print(f"ERROR: {health.detail}", file=sys.stderr)
                return EXIT_RUNTIME
            if not health.model_present:
                print(f"ERROR: model {settings.ollama_model!r} is not installed.", file=sys.stderr)
                print(f"Download it once: ollama pull {settings.ollama_model}", file=sys.stderr)
                print(f"Installed models: {', '.join(health.models) or '(none)'}", file=sys.stderr)
                return EXIT_RUNTIME

        results = await run_cases(brain, cases, runs=args.runs)
    finally:
        if not args.stub:
            await brain.aclose()  # type: ignore[attr-defined]

    print("=" * 78)
    print(render_report(results, label=label))
    print("=" * 78)

    if args.out:
        out_path = Path(args.out).expanduser()
    elif settings is not None:
        out_path = settings.resolved_data_dir / "eval" / f"eval-{_stamp()}.json"
    else:
        out_path = Path("runtime-data") / "eval" / f"eval-{_stamp()}.json"
    saved = _save(results, path=out_path, label=label)
    print(f"results saved: {saved}")

    return EXIT_OK if summarise(results)["all_stable"] else EXIT_CASE_FAILURES


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_model", description="Score the local model against fixed capability cases."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="case file (JSON)")
    parser.add_argument("--runs", type=int, default=3, help="attempts per case (default 3)")
    parser.add_argument("--model", help="override CLAIREAI_OLLAMA_MODEL")
    parser.add_argument("--base-url", help="override CLAIREAI_OLLAMA_BASE_URL (loopback only)")
    parser.add_argument("--out", help="write the JSON report here instead of the data directory")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="run with a stub brain to verify the harness itself (not a model measurement)",
    )
    return parser


def main(argv=None) -> int:  # noqa: ANN001
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        print("ERROR: --runs must be >= 1", file=sys.stderr)
        return EXIT_CONFIG
    try:
        return asyncio.run(_run(args))
    except ValidationError as exc:
        print("Invalid configuration:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  {field}: {error['msg']}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
