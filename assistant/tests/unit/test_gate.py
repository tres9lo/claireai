"""A candidate is installed only on a strict tool win with no chat regression."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from claireai.gate import CandidateGate, GateError


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    assistant = repo / "assistant"
    scripts = repo / "scripts"
    (assistant / "src" / "claireai").mkdir(parents=True)
    (assistant / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    scripts.mkdir()
    for name in ("tool_call_eval.py", "conversation_eval.py", "eval_model.py"):
        (scripts / name).write_text("# harness\n", encoding="utf-8")

    data = tmp_path / "data"
    installed = data / "models" / "claire-scratch.pt"
    candidate = data / "models" / "candidates" / "claire-curriculum-test.pt"
    for path, marker in ((installed, b"old"), (candidate, b"new")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(marker)
        path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    return assistant, data, candidate


def _runner(
    *,
    candidate_tools: int = 61,
    installed_tools: int = 60,
    candidate_chat: int = 14,
    installed_chat: int = 14,
    exit_code: int = 0,
):
    def run(command, cwd):  # noqa: ANN001, ANN202
        script = Path(command[1]).name
        checkpoint = Path(command[command.index("--model-checkpoint") + 1])
        is_candidate = checkpoint.parent.name == "candidates"
        if script == "tool_call_eval.py":
            payload = {
                "summary": {
                    "checks_passed": candidate_tools if is_candidate else installed_tools,
                    "checks": 165,
                    "cases_stable": 10,
                    "cases": 19,
                }
            }
        elif script == "conversation_eval.py":
            payload = {
                "passed": candidate_chat if is_candidate else installed_chat,
                "total": 21,
            }
        else:
            payload = {"summary": {"passed": 5, "total": 13}}
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=exit_code, stdout="", stderr="")

    return run


def test_a_tool_win_with_stable_conversation_is_installed_with_a_backup(tmp_path) -> None:
    assistant, data, candidate = _layout(tmp_path)
    gate = CandidateGate(data, checkout=assistant, runner=_runner())

    result = gate.review(candidate)

    assert result["passed"] is True
    assert result["promoted"] is True
    assert (data / "models" / "claire-scratch.pt").read_bytes() == b"new"
    assert (data / "models" / "claire-scratch.json").read_text(encoding="utf-8") == "{}\n"
    assert candidate.is_file()
    archives = list((data / "models" / "archive").glob("claire-scratch-*.pt"))
    assert archives and archives[0].read_bytes() == b"old"


def test_a_conversation_regression_rejects_an_otherwised_better_tool_score(tmp_path) -> None:
    assistant, data, candidate = _layout(tmp_path)
    gate = CandidateGate(
        data,
        checkout=assistant,
        runner=_runner(candidate_tools=70, candidate_chat=13),
    )

    result = gate.review(candidate)

    assert result["passed"] is False
    assert result["promoted"] is False
    assert "conversation regressed" in result["reason"]
    assert (data / "models" / "claire-scratch.pt").read_bytes() == b"old"
    assert not (data / "models" / "archive").exists()


def test_equal_tool_checks_reject_the_candidate(tmp_path) -> None:
    assistant, data, candidate = _layout(tmp_path)
    gate = CandidateGate(data, checkout=assistant, runner=_runner(candidate_tools=60))

    result = gate.review(candidate)

    assert result["passed"] is False
    assert "did not improve" in result["reason"]
    assert (data / "models" / "claire-scratch.pt").read_bytes() == b"old"


def test_a_failed_suite_never_promotes(tmp_path) -> None:
    assistant, data, candidate = _layout(tmp_path)
    gate = CandidateGate(
        data,
        checkout=assistant,
        runner=_runner(candidate_tools=165, exit_code=2),
    )

    with pytest.raises(GateError, match="exited 2"):
        gate.review(candidate)
    assert (data / "models" / "claire-scratch.pt").read_bytes() == b"old"


def test_missing_scripts_or_sidecars_refuse_to_compare(tmp_path) -> None:
    assistant, data, candidate = _layout(tmp_path)
    (assistant.parent / "scripts" / "eval_model.py").unlink()
    with pytest.raises(GateError, match="scripts are missing"):
        CandidateGate(data, checkout=assistant, runner=_runner()).review(candidate)

    candidate.with_suffix(".json").unlink()
    with pytest.raises(GateError, match="sidecar is missing"):
        CandidateGate(data, checkout=assistant, runner=_runner()).review(candidate)


def test_different_case_counts_are_not_compared(tmp_path) -> None:
    assistant, data, candidate = _layout(tmp_path)
    gate = CandidateGate(data, checkout=assistant, runner=_runner())
    scores = {
        "candidate": {
            "tools": {"checks_passed": 70, "checks": 166},
            "conversation": {"passed": 14, "total": 21},
        },
        "installed": {
            "tools": {"checks_passed": 60, "checks": 165},
            "conversation": {"passed": 14, "total": 21},
        },
    }

    passed, reason = gate.decide(scores["candidate"], scores["installed"])

    assert passed is False
    assert "case counts differ" in reason
