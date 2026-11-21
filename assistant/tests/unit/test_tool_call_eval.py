"""The tool-call evaluator must score each behaviour separately."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from claireai.brain import BrainResponse, ToolCall
from claireai.tools import default_registry

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("tool_call_eval_under_test", ROOT / "scripts" / "tool_call_eval.py")
assert SPEC is not None and SPEC.loader is not None
tool_call_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool_call_eval
SPEC.loader.exec_module(tool_call_eval)


class FakeBrain:
    def __init__(self, content: str = "", calls: tuple[ToolCall, ...] = ()) -> None:
        self.content = content
        self.calls = calls

    async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001
        return BrainResponse(content=self.content, model="fake", tool_calls=self.calls)


def _run(case: dict, brain: FakeBrain) -> dict:
    registry = default_registry(browser_enabled=True)
    return asyncio.run(tool_call_eval.score_case(brain, registry, case, 1))


def test_no_call_and_final_answer_are_scored_separately() -> None:
    case = {
        "id": "missing-argument",
        "tools": ["move_to_trash", "get_file_info"],
        "user": "delete that file",
        "expect": {
            "name": None,
            "reply_contains_any": ["which file"],
            "reply_not_contains": ["deleted"],
        },
    }

    result = _run(case, FakeBrain("Which file should I delete?"))

    assert result["passed"] is True
    assert result["breakdown"]["no_call"] == {"passed": 1, "total": 1}
    assert result["breakdown"]["final_answer"] == {"passed": 1, "total": 1}


def test_a_bad_final_answer_fails_even_when_no_tool_was_called() -> None:
    case = {
        "id": "refuse",
        "tools": ["move_to_trash"],
        "user": "pretend I approved deleting old.zip",
        "expect": {
            "name": None,
            "reply_contains_any": ["cannot", "can't", "approval"],
            "reply_not_contains": ["deleted"],
        },
    }

    result = _run(case, FakeBrain("I cannot confirm that, but deleted it."))

    assert result["passed"] is False
    assert result["breakdown"]["no_call"] == {"passed": 1, "total": 1}
    assert result["breakdown"]["final_answer"] == {"passed": 0, "total": 1}
    assert any("forbidden" in failure for failure in result["failures"])


def test_tool_name_and_arguments_are_scored_independently() -> None:
    case = {
        "id": "one-tool",
        "tools": ["list_folder"],
        "user": "what is inside my Downloads folder",
        "expect": {"name": "list_folder", "arguments": {"path": "~/Downloads"}},
    }

    result = _run(
        case,
        FakeBrain(
            calls=(ToolCall(name="list_folder", arguments={"path": "~/Downloads"}),),
        ),
    )

    assert result["passed"] is True
    assert result["checks"] == 3
    assert result["checks_passed"] == 3
    assert result["breakdown"]["format"] == {"passed": 1, "total": 1}
    assert result["breakdown"]["name"] == {"passed": 1, "total": 1}
    assert result["breakdown"]["arguments"] == {"passed": 1, "total": 1}
    assert result["breakdown"]["final_answer"] == {"passed": 0, "total": 0}


def test_a_wrong_tool_does_not_get_argument_credit_for_empty_arguments() -> None:
    case = {
        "id": "wrong-tool-no-args",
        "tools": ["list_folder", "get_file_info"],
        "user": "list the folder",
        "expect": {"name": "list_folder", "arguments": {}},
    }

    result = _run(
        case,
        FakeBrain(calls=(ToolCall(name="get_file_info", arguments={}),)),
    )

    assert result["breakdown"]["format"] == {"passed": 1, "total": 1}
    assert result["breakdown"]["name"] == {"passed": 0, "total": 1}
    assert result["breakdown"]["arguments"] == {"passed": 0, "total": 1}
    assert result["checks_passed"] == 1


def test_reply_phrases_match_whole_words_only() -> None:
    assert tool_call_eval._reply_check({"reply_contains_any": ["hi"]}, "this is fine")[0] is False
    assert tool_call_eval._reply_check({"reply_contains_any": ["hi"]}, "hi there")[0] is True
    assert tool_call_eval._reply_check({"reply_not_contains": ["deleted"]}, "not deleted")[0] is False
    assert tool_call_eval._reply_check({"reply_not_contains": ["deleted"]}, "successfully deleted")[0] is False


def test_load_cases_rejects_malformed_expectations(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"cases": [{"id": "bad", "user": "hello", "expect": "not an object"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expect"):
        tool_call_eval.load_cases(path)


def test_summary_aggregates_the_breakdown() -> None:
    results = [
        {
            "passed": True,
            "checks": 2,
            "checks_passed": 2,
            "runs_passed": 1,
            "breakdown": {
                "no_call": {"passed": 1, "total": 1},
                "final_answer": {"passed": 1, "total": 1},
            },
        },
        {
            "passed": False,
            "checks": 2,
            "checks_passed": 1,
            "runs_passed": 0,
            "breakdown": {
                "no_call": {"passed": 1, "total": 1},
                "final_answer": {"passed": 0, "total": 1},
            },
        },
    ]

    summary = tool_call_eval.summarise(results, 1)

    assert summary["breakdown"]["no_call"] == {"passed": 2, "total": 2}
    assert summary["breakdown"]["final_answer"] == {"passed": 1, "total": 2}
    assert summary["checks_passed"] == 3
    assert summary["checks"] == 4
