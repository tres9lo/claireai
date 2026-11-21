"""The training catalog must be explicit about licence and access."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from claire_brain.tokenizer import BytePairTokenizer

ROOT = Path(__file__).resolve().parents[3]
ALLOWED_ACCESS = {"public", "request", "credentials", "do-not-use"}
FETCHER = ROOT / "brain-lab" / "fetch_corpora.py"
MERGER = ROOT / "brain-lab" / "merge_corpora.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetcher_module():
    return _module(FETCHER, "fetch_corpora_under_test")


def _merger_module():
    return _module(MERGER, "merge_corpora_under_test")


def test_every_source_has_a_licence_link_and_access_state() -> None:
    payload = json.loads((ROOT / "brain-lab" / "data-sources.json").read_text(encoding="utf-8"))
    sources = payload["sources"]

    assert len(sources) >= 10
    assert len({source["id"] for source in sources}) == len(sources)
    for source in sources:
        assert source["name"]
        assert source["url"].startswith("https://")
        assert source["license"]
        assert source["access"] in ALLOWED_ACCESS
        assert source["domains"]


def test_sources_with_restrictions_are_not_marked_for_training() -> None:
    payload = json.loads((ROOT / "brain-lab" / "data-sources.json").read_text(encoding="utf-8"))
    by_id = {source["id"]: source for source in payload["sources"]}

    assert by_id["stack-exchange"]["access"] == "do-not-use"
    assert by_id["enron-email"]["access"] == "do-not-use"
    assert by_id["fineweb-edu"]["license"].startswith("ODC-By")


def test_toolace_call_syntax_is_normalised_without_guessing() -> None:
    fetcher = _fetcher_module()
    calls = fetcher._parse_toolace_calls(
        '[GetCompetitions(), Get Manager Placeholder Image(sport="football")]'
    )
    assert calls == [
        {"name": "GetCompetitions", "arguments": {}},
        {"name": "Get Manager Placeholder Image", "arguments": {"sport": "football"}},
    ]
    assert fetcher._parse_toolace_calls("[Broken(name=)]") is None


def test_hermes_call_blocks_are_parsed() -> None:
    fetcher = _fetcher_module()
    calls = fetcher._parse_hermes_calls(
        '<tool_call>\n{"name":"list_cameras","arguments":{"include_offline":false}}\n</tool_call>'
    )
    assert calls == [{"name": "list_cameras", "arguments": {"include_offline": False}}]
    nested = fetcher._parse_hermes_calls(
        '<tool_call>{"name":"save","arguments":{"payload":{"a":1,"b":2}}}</tool_call>'
    )
    assert nested == [{"name": "save", "arguments": {"payload": {"a": 1, "b": 2}}}]
    assert fetcher._parse_hermes_calls("<tool_call>{not json}</tool_call>") is None


class _BudgetTokenizer:
    def encode(self, text: str, *, append_eos: bool = True) -> list[int]:
        return [1] * (200 if "over budget" in text else 10)


def test_tool_records_use_the_real_tokenizer_budget() -> None:
    fetcher = _fetcher_module()
    values = dict(
        source="toolace",
        license_name="Apache-2.0",
        url="https://example.test/source",
        available_tools=["list_folder"],
        tokenizer=_BudgetTokenizer(),
        max_tokens=64,
    )
    record = fetcher._tool_record("hello there", "short answer", **values)
    assert record is not None
    assert "list_folder" in record["system"]
    assert "TOOLS:" in record["system"]
    assert fetcher._tool_record("hello there", "over budget answer", **values) is None


def test_a_non_object_tokenizer_manifest_is_rejected(tmp_path: Path) -> None:
    fetcher = _fetcher_module()
    manifest = tmp_path / "model.json"
    manifest.write_text("[]", encoding="utf-8")

    try:
        fetcher._load_tokenizer(manifest)
    except fetcher.FetchError as exc:
        assert "not an object" in str(exc)
    else:
        raise AssertionError("a list manifest must be rejected")


def test_a_tokenizer_manifest_is_loaded_from_its_merges(tmp_path: Path) -> None:
    fetcher = _fetcher_module()
    tokenizer = BytePairTokenizer.train(["hello hello hello"], vocab_size=260)
    manifest = tmp_path / "model.json"
    manifest.write_text(
        json.dumps({"tokenizer_merges": [list(pair) for pair in tokenizer.merges]}),
        encoding="utf-8",
    )

    loaded = fetcher._load_tokenizer(manifest)
    assert loaded.merges == tokenizer.merges


def test_tool_eval_prompts_are_held_out_everywhere() -> None:
    prompt = "what is inside my Downloads folder"
    assert prompt in _fetcher_module()._held_out_prompts()
    assert prompt in _merger_module().held_out_prompts()


def test_toolace_fetch_keeps_only_known_one_or_two_call_rows(monkeypatch) -> None:
    fetcher = _fetcher_module()
    system = (
        "You are an expert in composing functions.\n"
        "Here is a list of functions in JSON format that you can invoke:\n"
        '[{"name": "GetCompetitions"}, {"name": "Get Manager Placeholder Image"}]'
    )
    good = {
        "system": system,
        "conversations": [
            {"from": "user", "value": "Show me the televised football competitions."},
            {"from": "assistant", "value": "[GetCompetitions()]"},
        ],
    }
    unknown = {
        "system": system,
        "conversations": [
            {"from": "user", "value": "Call a tool that is not in this schema."},
            {"from": "assistant", "value": "[NotATool()]"},
        ],
    }
    monkeypatch.setattr(fetcher, "_rows", lambda *args, **kwargs: [good, unknown])

    records = fetcher.fetch_toolace(10)
    assert len(records) == 1
    assert records[0]["source"] == "toolace"
    assert records[0]["available_tools"] == ["GetCompetitions", "Get Manager Placeholder Image"]
    assert records[0]["messages"][1]["content"] == (
        '<tool_call>{"name":"GetCompetitions","arguments":{}}</tool_call>'
    )


def test_hermes_fetch_keeps_a_cited_tool_schema(monkeypatch) -> None:
    fetcher = _fetcher_module()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_cameras",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    row = {
        "tools": json.dumps(tools),
        "conversations": [
            {"from": "system", "value": "Use the supplied functions."},
            {"from": "human", "value": "List every camera on my home network."},
            {
                "from": "gpt",
                "value": '<tool_call>{"name":"list_cameras","arguments":{"include_offline":false}}</tool_call>',
            },
        ],
    }
    monkeypatch.setattr(fetcher, "_rows", lambda *args, **kwargs: [row])

    records = fetcher.fetch_hermes(10)
    assert len(records) == 1
    assert records[0]["source"] == "hermes-function-calling-v1"
    assert records[0]["available_tools"] == ["list_cameras"]
    assert "<tool_call>" in records[0]["messages"][1]["content"]
