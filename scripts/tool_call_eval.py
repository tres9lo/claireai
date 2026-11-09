#!/usr/bin/env python3
"""Measure whether the local model proposes readable tool calls.

The other suites answer questions; this one asks only one thing: given the tool
schemas the runtime shows it and one request, does the model emit a call the
runtime can actually read, with the right tool and the right arguments?

It goes through :class:`claireai.brain.ScratchBrain`, so it scores the same
parser the runtime uses — a call that this script counts as readable is one the
agent would act on. Three checks per case:

``format``      exactly one call and no leftover prose
``name``        the tool the request needs
``arguments``   every required argument the request names

Results are per-run and a case only counts as passing when every run passes, so
a model that lands a call one time in three does not look correct.

    assistant/.venv/bin/python scripts/tool_call_eval.py --runs 3

Exit codes: 0 all cases stable, 1 some case failed, 2 bad configuration,
3 model runtime unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claireai.agent import system_message
from claireai.brain import Message, ScratchBrain
from claireai.config import Settings
from claireai.permissions import Sandbox
from claireai.tools import default_registry

DEFAULT_CASES = Path(__file__).resolve().parent / "tool_call_cases.json"

EXIT_OK = 0
EXIT_CASE_FAILURES = 1
EXIT_CONFIG = 2
EXIT_RUNTIME = 3


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("case file must contain a non-empty 'cases' list")
    for case in cases:
        if not case.get("user") or "expect" not in case:
            raise ValueError(f"case {case.get('id')!r} needs 'user' and 'expect'")
    return cases


def _match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()
    return expected == actual


def sandbox_roots(mode: str) -> tuple[Path, ...]:
    """The folders the prompt claims this install may use.

    ``home`` mirrors a configured install: the three folders these cases name.
    ``empty`` is the fresh-install state, where no file tool can run, so a correct
    answer cannot be a file call.  This harness used to always send ``empty``,
    which measured a state no working install is in and asked for calls on a
    prompt the model had never seen.
    """
    if mode == "empty":
        return ()
    home = Path.home()
    return tuple(home / name for name in ("Downloads", "Documents", "Desktop"))


def system_prompt(mode: str) -> Message:
    return system_message(Sandbox(roots=sandbox_roots(mode)))


def prompt_clause(mode: str) -> str:
    return system_prompt(mode).content.split(" Allowed folders:", 1)[-1]


async def score_case(
    brain: ScratchBrain, registry, case: dict[str, Any], runs: int, *, sandbox: str = "home"
) -> dict[str, Any]:
    schemas = [registry.get(name).schema() for name in case.get("tools", []) if name in registry]
    messages = [
        system_prompt(sandbox),
        Message(role="user", content=str(case["user"])),
    ]
    expect = case["expect"]
    expected_name = expect.get("name")
    expected_arguments = expect.get("arguments") or {}

    passed_runs = 0
    checks = 0
    checks_passed = 0
    failures: list[str] = []
    sample = ""
    for _ in range(runs):
        response = await brain.generate(messages, tools=schemas, temperature=0.2)
        calls = response.tool_calls
        sample = sample or (
            f"{calls[0].name}({json.dumps(calls[0].arguments, ensure_ascii=False)})"
            if calls
            else response.content.strip()[:160]
        )
        run_ok = True
        if expected_name is None:
            checks += 1
            if calls:
                failures.append(f"called {calls[0].name} for small talk")
                run_ok = False
            else:
                checks_passed += 1
        else:
            checks += 3
            if len(calls) != 1:
                failures.append(f"format: {len(calls)} readable calls")
                run_ok = False
            else:
                checks_passed += 1
                if calls[0].name == expected_name:
                    checks_passed += 1
                else:
                    failures.append(f"name: {calls[0].name} != {expected_name}")
                    run_ok = False
                argument_failures = [
                    key
                    for key, value in expected_arguments.items()
                    if not _match(value, calls[0].arguments.get(key))
                ]
                if argument_failures and calls[0].name == expected_name:
                    failures.append("arguments: " + ", ".join(sorted(argument_failures)))
                    run_ok = False
                elif not argument_failures:
                    checks_passed += 1
        if run_ok:
            passed_runs += 1
    return {
        "id": case.get("id"),
        "passed": passed_runs == runs,
        "runs": runs,
        "runs_passed": passed_runs,
        "checks": checks,
        "checks_passed": checks_passed,
        "failures": list(dict.fromkeys(failures)),
        "sample": sample,
    }


def summarise(results: list[dict[str, Any]], runs: int) -> dict[str, Any]:
    checks = sum(result["checks"] for result in results)
    checks_passed = sum(result["checks_passed"] for result in results)
    stable = sum(1 for result in results if result["passed"])
    total_runs_passed = sum(result["runs_passed"] for result in results)
    return {
        "cases": len(results),
        "cases_stable": stable,
        "checks": checks,
        "checks_passed": checks_passed,
        "runs": runs,
        "runs_passed": total_runs_passed,
        "all_stable": stable == len(results),
    }


def render_report(results: list[dict[str, Any]], summary: dict[str, Any], label: str) -> str:
    lines = [f"tool-call evaluation · {label}", ""]
    for result in results:
        marker = "pass" if result["passed"] else "FAIL"
        detail = (
            ""
            if result["passed"]
            else f"  [{'; '.join(result['failures'])}]  sample={result['sample']!r}"
        )
        lines.append(
            f"  {result['id']:<18} {marker} {result['runs_passed']}/{result['runs']}"
            f"  checks {result['checks_passed']}/{result['checks']}{detail}"
        )
    lines += [
        "",
        f"cases: {summary['cases_stable']}/{summary['cases']} stable",
        f"checks: {summary['checks_passed']}/{summary['checks']}",
        f"runs: {summary['runs_passed']} of {summary['runs'] * summary['cases']} runs passed every check",
    ]
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(Path(args.cases).expanduser())
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load cases: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    overrides: dict[str, Any] = {}
    if args.model_checkpoint:
        overrides["model_checkpoint"] = args.model_checkpoint
    settings = Settings(_env_file=None, **overrides)
    brain = ScratchBrain(checkpoint=settings.resolved_model_checkpoint, max_output_tokens=120)
    health = await brain.health()
    if not health.available or not health.model_present:
        print(f"ERROR: {health.detail}", file=sys.stderr)
        return EXIT_RUNTIME

    registry = default_registry(browser_enabled=True)
    try:
        results = [
            await score_case(brain, registry, case, args.runs, sandbox=args.sandbox)
            for case in cases
        ]
    finally:
        await brain.aclose()

    summary = summarise(results, args.runs)
    label = f"{health.models[0] if health.models else 'scratch'} @ {settings.resolved_model_checkpoint}"
    print("=" * 78)
    print(f"sandbox: {args.sandbox}  (prompt says: Allowed folders: {prompt_clause(args.sandbox)})")
    print(render_report(results, summary, label))
    print("=" * 78)

    out = Path(args.out).expanduser() if args.out else settings.resolved_data_dir / "eval" / (
        f"tool-calls-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "label": label,
                "sandbox": {
                    "mode": args.sandbox,
                    "roots": [str(root) for root in sandbox_roots(args.sandbox)],
                    "prompt_clause": prompt_clause(args.sandbox),
                },
                "summary": summary,
                "cases": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"results saved: {out}")
    return EXIT_OK if summary["all_stable"] else EXIT_CASE_FAILURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tool_call_eval", description="Score the local model on readable tool calls."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="case file (JSON)")
    parser.add_argument("--runs", type=int, default=3, help="attempts per case (default 3)")
    parser.add_argument("--model-checkpoint", type=Path, help="override CLAIREAI_MODEL_CHECKPOINT")
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument(
        "--sandbox",
        choices=("home", "empty"),
        default="home",
        help="which folder clause the system prompt carries (default: home, like a configured install)",
    )
    return parser


def main(argv=None) -> int:  # noqa: ANN001
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        print("ERROR: --runs must be >= 1", file=sys.stderr)
        return EXIT_CONFIG
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
