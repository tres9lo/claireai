"""The post-training gate must run every fixed suite and fail closed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("run_gate_under_test", ROOT / "scripts" / "run_gate.py")
assert SPEC is not None and SPEC.loader is not None
run_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_gate
SPEC.loader.exec_module(run_gate)


def test_gate_includes_model_conversation_and_both_tool_suites(tmp_path: Path) -> None:
    stages = run_gate.build_stages(
        tmp_path / "candidate.pt",
        tmp_path / "gate",
        python="/venv/python",
        runs=2,
        conversation_runs=1,
    )

    assert [stage.name for stage in stages] == [
        "model",
        "conversation",
        "tool-calls",
        "tool-calls-extended",
    ]
    assert stages[2].command[-4:] == [
        "--cases",
        str(ROOT / "scripts" / "tool_call_cases.json"),
        "--out",
        str(tmp_path / "gate" / "tool-calls.json"),
    ]
    assert str(ROOT / "scripts" / "tool_call_cases_extended.json") in stages[3].command
    assert stages[0].command.count("--runs") == 1
    assert stages[0].command[stages[0].command.index("--runs") + 1] == "2"


def test_gate_stops_after_a_failure_when_requested(tmp_path: Path) -> None:
    stages = [
        run_gate.Stage("first", ["first"], tmp_path / "first.json"),
        run_gate.Stage("second", ["second"], tmp_path / "second.json"),
    ]
    calls = []

    def runner(command, *, cwd, check):  # noqa: ANN001
        calls.append(command)
        return SimpleNamespace(returncode=1 if command == ["first"] else 0)

    results = run_gate.run_stages(stages, runner=runner, fail_fast=True)

    assert len(results) == 1
    assert results[0]["passed"] is False
    assert calls == [["first"]]


def test_gate_dry_run_is_json_and_does_not_execute(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "gate"

    status = run_gate.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(output),
            "--dry-run",
            "--python",
            "/venv/python",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output.resolve())
    assert [stage["name"] for stage in payload["stages"]] == [
        "model",
        "conversation",
        "tool-calls",
        "tool-calls-extended",
    ]
    assert not output.exists()
