"""The corpus audit must catch budget, call and contamination mistakes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BRAIN_LAB = ROOT / "brain-lab"
if str(BRAIN_LAB) not in sys.path:
    sys.path.insert(0, str(BRAIN_LAB))

import audit_corpus  # noqa: E402


class FakeTokenizer:
    def encode(self, text: str, *, append_eos: bool = True) -> list[int]:
        values = [1] * len(text.split())
        return values + ([0] if append_eos else [])


def _record(user: str, assistant: str, *, available: list[str] | None = None) -> dict:
    return {
        "system": "You are ClaireAI. TOOLS: You may call list_folder.",
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        "available_tools": available or ["list_folder"],
    }


def test_audit_counts_tokens_calls_unknowns_and_duplicates() -> None:
    good = _record(
        "list my downloads",
        '<tool_call>{"name":"list_folder","arguments":{"path":"~/Downloads"}}</tool_call>',
    )
    unknown = _record(
        "trade for me",
        '<tool_call>{"name":"execute_trades","arguments":{}}</tool_call>',
    )
    malformed = _record("read this", "<tool_call>not json</tool_call>")
    long_target = _record("write a lot", "word " * 200)
    two_calls = _record(
        "list and move",
        '<tool_call>{"name":"list_folder","arguments":{"path":"~/Downloads"}}</tool_call>'
        '<tool_call>{"name":"list_folder","arguments":{"path":"~/Documents"}}</tool_call>',
    )

    report, pairs = audit_corpus.audit_records(
        [good, good, unknown, malformed, long_target, two_calls],
        FakeTokenizer(),
        context=512,
        target_cap=192,
        held_out=["list my downloads"],
        local_tools=["list_folder"],
    )

    assert report["records"] == 6
    assert report["over_target"] == 1
    assert report["unknown_tools"] == {"execute_trades": 1}
    assert report["malformed_tool_calls"] == 1
    assert report["duplicate_records"] == 1
    assert report["held_out_overlap"] == 2
    assert report["calls"]["2"] == 1
    assert len(pairs) == 5


def test_audit_reports_missing_assistant_and_tool_block() -> None:
    record = {
        "system": "plain system",
        "messages": [{"role": "user", "content": "hello"}],
    }
    report, _ = audit_corpus.audit_records([record], FakeTokenizer(), local_tools=[])
    assert report["missing_tool_block"] == 1
    assert report["missing_assistant"] == 1


def test_corpus_files_prefers_train_and_validation(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "other.jsonl").write_text("", encoding="utf-8")

    assert [path.name for path in audit_corpus._corpus_files(tmp_path)] == [
        "train.jsonl",
        "validation.jsonl",
    ]
