"""Tests for the Ollama adapter.

These use httpx.MockTransport: they verify our request shape and error mapping,
NOT that a real Ollama install answers. The real check is the manual smoke test
recorded in PROGRESS.md.
"""

from __future__ import annotations

import json

import httpx
import pytest

from claireai.brain import (
    BrainResponseError,
    BrainUnavailable,
    Message,
    OllamaBrain,
    ToolCall,
)

BASE_URL = "http://127.0.0.1:11434"


def _brain(handler) -> OllamaBrain:
    client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return OllamaBrain(base_url=BASE_URL, model="qwen2.5:1.5b", client=client)


async def test_generate_sends_non_streaming_chat_request() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:1.5b",
                "message": {"role": "assistant", "content": "Hello there."},
                "prompt_eval_count": 11,
                "eval_count": 5,
            },
        )

    brain = _brain(handler)
    response = await brain.generate([Message(role="user", content="hi")])

    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "qwen2.5:1.5b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert response.content == "Hello there."
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 5


async def test_a_per_call_temperature_is_sent_without_changing_the_adapter() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    brain = OllamaBrain(
        base_url=BASE_URL, model="qwen2.5:1.5b", client=client, options={"num_predict": 40}
    )

    await brain.generate([Message(role="user", content="hi")], temperature=0.2)
    await brain.generate([Message(role="user", content="hi")], temperature=0.8)
    await brain.generate([Message(role="user", content="hi")])

    assert bodies[0]["options"] == {"num_predict": 40, "temperature": 0.2}
    assert bodies[1]["options"] == {"num_predict": 40, "temperature": 0.8}
    # No per-call temperature means the adapter's own options, untouched.
    assert bodies[2]["options"] == {"num_predict": 40}
    assert brain.options == {"num_predict": 40}


async def test_tools_are_forwarded_when_provided() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "message": {"content": "ok"}})

    tools = [{"type": "function", "function": {"name": "list_folder"}}]
    await _brain(handler).generate([Message(role="user", content="hi")], tools=tools)
    assert captured["tools"] == tools


async def test_decoding_options_are_forwarded() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "message": {"content": "ok"}})

    client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    brain = OllamaBrain(base_url=BASE_URL, model="m", client=client, options={"temperature": 0})
    await brain.generate([Message(role="user", content="hi")])
    assert captured["options"] == {"temperature": 0}


async def test_missing_model_error_names_the_pull_command() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(BrainResponseError, match="ollama pull qwen2.5:1.5b"):
        await _brain(handler).generate([Message(role="user", content="hi")])


async def test_server_error_is_reported_with_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(BrainResponseError, match="HTTP 500"):
        await _brain(handler).generate([Message(role="user", content="hi")])


async def test_connection_failure_raises_unavailable_with_action() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BrainUnavailable, match="ollama serve"):
        await _brain(handler).generate([Message(role="user", content="hi")])


async def test_unexpected_payload_shape_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(BrainResponseError, match="unexpected"):
        await _brain(handler).generate([Message(role="user", content="hi")])


async def test_health_reports_model_presence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen2.5:1.5b"}, {"name": "llama3.2:1b"}]},
        )

    health = await _brain(handler).health()
    assert health.available is True
    assert health.model_present is True
    assert "llama3.2:1b" in health.models


async def test_health_reports_missing_model_without_failing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.2:1b"}]})

    health = await _brain(handler).health()
    assert health.available is True
    assert health.model_present is False


async def test_health_reports_unreachable_runtime() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    health = await _brain(handler).health()
    assert health.available is False
    assert "cannot reach Ollama" in health.detail


# --- tool calling (https://docs.ollama.com/capabilities/tool-calling) ---------


def _tool_reply(message: dict) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "qwen2.5:1.5b", "message": message})

    return handler


async def test_tool_calls_are_parsed_from_the_reply() -> None:
    handler = _tool_reply(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "index": 0,
                        "name": "list_folder",
                        "arguments": {"path": "/home/tresor/Downloads"},
                    },
                }
            ],
        }
    )

    response = await _brain(handler).generate([Message(role="user", content="list it")])

    assert response.content == ""
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "list_folder"
    assert call.arguments == {"path": "/home/tresor/Downloads"}
    assert call.index == 0


async def test_tool_call_arguments_sent_as_a_json_string_are_parsed() -> None:
    handler = _tool_reply(
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_files", "arguments": '{"pattern": "*.pdf"}'}}
            ],
        }
    )

    response = await _brain(handler).generate([Message(role="user", content="find pdfs")])
    assert response.tool_calls[0].arguments == {"pattern": "*.pdf"}


async def test_content_may_be_absent_alongside_a_tool_call() -> None:
    handler = _tool_reply({"tool_calls": [{"function": {"name": "list_folder", "arguments": {}}}]})
    response = await _brain(handler).generate([Message(role="user", content="go")])
    assert response.content == ""
    assert response.tool_calls[0].name == "list_folder"


async def test_a_tool_call_without_a_name_is_rejected() -> None:
    handler = _tool_reply({"content": "", "tool_calls": [{"function": {"arguments": {}}}]})
    with pytest.raises(BrainResponseError, match="without a name"):
        await _brain(handler).generate([Message(role="user", content="go")])


async def test_tool_call_arguments_that_are_not_json_are_rejected() -> None:
    handler = _tool_reply(
        {"content": "", "tool_calls": [{"function": {"name": "list_folder", "arguments": "{not json"}}]}
    )
    with pytest.raises(BrainResponseError, match="not valid JSON"):
        await _brain(handler).generate([Message(role="user", content="go")])


async def test_tool_call_arguments_that_are_not_an_object_are_rejected() -> None:
    handler = _tool_reply(
        {"content": "", "tool_calls": [{"function": {"name": "list_folder", "arguments": [1, 2]}}]}
    )
    with pytest.raises(BrainResponseError, match="not an object"):
        await _brain(handler).generate([Message(role="user", content="go")])


async def test_assistant_tool_calls_and_tool_results_are_serialized_for_the_wire() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "message": {"content": "done"}})

    call = ToolCall(name="list_folder", arguments={"path": "/tmp"}, index=0)
    await _brain(handler).generate(
        [
            Message(role="user", content="list /tmp"),
            Message(role="assistant", content="", tool_calls=(call,)),
            Message(role="tool", tool_name="list_folder", content="2 entries"),
        ]
    )

    messages = captured["messages"]
    assert messages[1]["tool_calls"] == [
        {"type": "function", "function": {"index": 0, "name": "list_folder", "arguments": {"path": "/tmp"}}}
    ]
    assert messages[2] == {"role": "tool", "tool_name": "list_folder", "content": "2 entries"}
    # A plain message must not grow empty tool fields.
    assert messages[0] == {"role": "user", "content": "list /tmp"}
