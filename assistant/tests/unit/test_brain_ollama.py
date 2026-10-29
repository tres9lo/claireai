"""Tests for the Ollama adapter.

These use httpx.MockTransport: they verify our request shape and error mapping,
NOT that a real Ollama install answers. The real check is the manual smoke test
recorded in PROGRESS.md.
"""

from __future__ import annotations

import json

import httpx
import pytest

from claireai.brain import BrainResponseError, BrainUnavailable, Message, OllamaBrain

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


async def test_tools_are_forwarded_when_provided() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "message": {"content": "ok"}})

    tools = [{"type": "function", "function": {"name": "list_folder"}}]
    await _brain(handler).generate([Message(role="user", content="hi")], tools=tools)
    assert captured["tools"] == tools


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
