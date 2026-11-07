"""The tolerant tool-call reader, and what it deliberately refuses to repair."""

from __future__ import annotations

import pytest

from claireai.brain.tool_calls import looks_like_tool_call, parse_tool_calls

OFFERED = ("list_folder", "read_text_file", "move_file")


def only(text: str, offered=OFFERED):  # noqa: ANN001, ANN201
    parsed = parse_tool_calls(text, offered)
    assert parsed.ok, parsed.problem
    assert len(parsed.calls) == 1
    return parsed.calls[0]


def test_the_canonical_shape_is_read() -> None:
    call = only('<tool_call>{"name":"list_folder","arguments":{"path":"~/Downloads"}}</tool_call>')
    assert call.name == "list_folder"
    assert call.arguments == {"path": "~/Downloads"}


def test_a_missing_closing_tag_is_repaired() -> None:
    # Measured failure: the model stops after the object, so the tag that opens
    # the call is present and the one that closes it is not.
    call = only('<tool_call>{"name":"list_folder","arguments":{"path":"~/Downloads"}}')
    assert call.name == "list_folder"


def test_a_fenced_payload_is_read() -> None:
    call = only('```json\n{"name":"read_text_file","arguments":{"path":"~/a.txt"}}\n```')
    assert call.name == "read_text_file"


def test_a_bare_payload_is_read() -> None:
    call = only('{"name":"move_file","arguments":{"source":"~/a","destination":"~/b"}}')
    assert call.name == "move_file"


def test_the_wrong_key_the_model_keeps_producing_is_repaired() -> None:
    # This is the exact shape that used to kill the task with "malformed
    # tool-call JSON": a "tool" key where "name" belongs.
    call = only('{"tool": "list_folder", "arguments": {"path": "~/Downloads"}}')
    assert call.name == "list_folder"


def test_single_quotes_and_a_trailing_comma_are_repaired() -> None:
    call = only("<tool_call>{'name': 'list_folder', 'arguments': {'path': '~/Downloads',},}</tool_call>")
    assert call.arguments == {"path": "~/Downloads"}


def test_arguments_sent_as_a_json_string_are_decoded() -> None:
    call = only('<tool_call>{"name":"list_folder","arguments":"{\\"path\\":\\"~/Downloads\\"}"}</tool_call>')
    assert call.arguments == {"path": "~/Downloads"}


def test_the_openai_nested_function_shape_is_read() -> None:
    call = only('{"type":"function","function":{"name":"list_folder","arguments":{"path":"~/"}}}')
    assert call.name == "list_folder"


def test_a_call_without_arguments_gets_an_empty_object() -> None:
    assert only('<tool_call>{"name":"list_folder"}</tool_call>').arguments == {}
    # With no tools offered, nothing is filtered: the agent's registry still
    # validates the name and the arguments.
    assert only('<tool_call>{"name":"get_page_text"}</tool_call>', ()).arguments == {}


def test_two_calls_in_one_message_are_both_read() -> None:
    parsed = parse_tool_calls(
        '<tool_call>{"name":"list_folder","arguments":{}}</tool_call>'
        '<tool_call>{"name":"read_text_file","arguments":{"path":"~/a"}}</tool_call>',
        OFFERED,
    )
    assert [call.name for call in parsed.calls] == ["list_folder", "read_text_file"]


def test_a_name_is_normalised_but_not_invented() -> None:
    assert only('<tool_call>{"name":"List Folder","arguments":{}}</tool_call>').name == "list_folder"
    assert only('<tool_call>{"name":"list-folder","arguments":{}}</tool_call>').name == "list_folder"


def test_an_unoffered_name_is_refused() -> None:
    parsed = parse_tool_calls('<tool_call>{"name":"rm_rf","arguments":{}}</tool_call>', OFFERED)
    assert parsed.calls == ()
    assert parsed.saw_intent is True
    assert "unknown tool" in parsed.problem


def test_a_name_the_runtime_never_offered_is_not_matched_to_a_similar_one() -> None:
    # "list_folders" must not silently become "list_folder": the model has to
    # learn the real name, and the repair turn teaches it.
    parsed = parse_tool_calls('<tool_call>{"name":"list_folders","arguments":{}}</tool_call>', OFFERED)
    assert parsed.calls == ()


def test_arguments_that_are_not_an_object_are_refused() -> None:
    parsed = parse_tool_calls('<tool_call>{"name":"list_folder","arguments":42}</tool_call>', OFFERED)
    assert parsed.calls == ()
    assert "not an object" in parsed.problem


def test_a_bare_object_inside_a_sentence_is_not_a_call() -> None:
    reply = 'You can send {"name": "list_folder"} to list a folder, for example.'
    assert parse_tool_calls(reply, OFFERED).calls == ()
    assert looks_like_tool_call(reply) is False


def test_a_tag_inside_prose_is_still_a_call() -> None:
    reply = 'Sure. <tool_call>{"name":"list_folder","arguments":{"path":"~/"}}</tool_call>'
    assert parse_tool_calls(reply, OFFERED).calls[0].name == "list_folder"


def test_plain_prose_is_not_an_attempted_call() -> None:
    assert looks_like_tool_call("Your Downloads folder contains two files.") is False
    assert looks_like_tool_call('{"tool": "list_folder", "arguments": {}}') is True
    assert looks_like_tool_call('<tool_call>{"name":"x"}') is True
    assert parse_tool_calls("Your Downloads folder contains two files.", OFFERED).saw_intent is False


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"list_folder"</tool_call>',
        '<tool_call>{"name":"list_folder","arguments":{"path":}}</tool_call>',
        '<tool_call>not json at all</tool_call>',
    ],
)
def test_unreadable_payloads_are_reported_rather_than_raised(text: str) -> None:
    parsed = parse_tool_calls(text, OFFERED)
    assert parsed.calls == ()
    assert parsed.saw_intent is True
    assert parsed.problem
