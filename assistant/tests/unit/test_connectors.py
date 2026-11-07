"""Connector tests against a transport stub.

No live API keys are used here or anywhere in the repository. The adapters are
exercised through ``httpx.MockTransport``, which means these tests prove the wire
format and the parsing, not that a given vendor is reachable today. ``claireai
doctor`` is the check that answers that question, against whatever credentials
the owner has configured.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from claireai.brain import (
    AnthropicBrain,
    BrainResponseError,
    BrainUnavailable,
    Message,
    OllamaBrain,
    OpenAIBrain,
    ScratchBrain,
    ToolCall,
    build_brain,
)
from claireai.config import Settings


def _brain(cls, handler, **kwargs):  # noqa: ANN001, ANN202
    kwargs.setdefault("api_key", "sk-secret-value")
    brain = cls(model="test-model", **kwargs)
    brain._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return brain


# -- configuration rules ----------------------------------------------------


def test_scratch_is_the_default_provider() -> None:
    settings = Settings(_env_file=None)
    assert settings.brain_provider == "scratch"
    assert settings.brain_api_key_value is None


def test_openai_and_anthropic_require_a_key() -> None:
    for provider in ("openai", "anthropic"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, brain_provider=provider, brain_model="some-model")


def test_a_connector_requires_a_model_name() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, brain_provider="ollama")


def test_plain_http_is_refused_off_loopback() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            brain_provider="openai",
            brain_model="m",
            brain_api_key="k",
            brain_base_url="http://api.example.com/v1",
        )


def test_loopback_http_is_allowed_and_defaulted() -> None:
    settings = Settings(_env_file=None, brain_provider="ollama", brain_model="llama3.2")
    assert settings.resolved_brain_base_url == "http://127.0.0.1:11434/api"

    local = Settings(
        _env_file=None,
        brain_provider="openai",
        brain_model="m",
        brain_api_key="k",
        brain_base_url="http://127.0.0.1:8000/v1",
    )
    assert local.resolved_brain_base_url == "http://127.0.0.1:8000/v1"


def test_the_api_key_is_not_in_the_settings_repr() -> None:
    settings = Settings(
        _env_file=None, brain_provider="openai", brain_model="m", brain_api_key="sk-live-1234"
    )
    assert "sk-live-1234" not in repr(settings)
    assert settings.brain_api_key_value == "sk-live-1234"


# -- factory ----------------------------------------------------------------


def test_factory_builds_the_scratch_brain_by_default(tmp_path) -> None:  # noqa: ANN001
    brain = build_brain(Settings(_env_file=None, data_dir=tmp_path))
    assert isinstance(brain, ScratchBrain)


def test_factory_builds_the_selected_connector() -> None:
    settings = Settings(_env_file=None, brain_provider="ollama", brain_model="llama3.2")
    brain = build_brain(settings)
    assert isinstance(brain, OllamaBrain)
    assert brain.base_url == "http://127.0.0.1:11434/api"


def test_factory_refuses_an_unknown_provider() -> None:
    with pytest.raises(ValueError):
        build_brain(SimpleNamespace(brain_provider="mystery"))


# -- OpenAI shape -----------------------------------------------------------


async def test_openai_sends_schemas_and_parses_a_call() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "list_folder",
                                        "arguments": '{"path": "~/Downloads"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        )

    brain = _brain(OpenAIBrain, handler)
    response = await brain.generate(
        [Message(role="system", content="you are ClaireAI"), Message(role="user", content="list")],
        tools=[{"name": "list_folder", "description": "list a folder", "parameters": {"type": "object"}}],
        temperature=0.2,
    )
    await brain.aclose()

    assert response.tool_calls[0].name == "list_folder"
    assert response.tool_calls[0].arguments == {"path": "~/Downloads"}
    assert response.prompt_tokens == 12
    payload = seen[0]
    assert payload["messages"][0] == {"role": "system", "content": "you are ClaireAI"}
    assert payload["tools"][0]["function"]["name"] == "list_folder"
    assert payload["tool_choice"] == "auto"
    assert payload["temperature"] == 0.2


async def test_openai_pairs_each_result_with_its_call_id() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    brain = _brain(OpenAIBrain, handler)
    await brain.generate(
        [
            Message(role="user", content="two things"),
            Message(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall(name="list_folder", arguments={"path": "~/Downloads"}),
                    ToolCall(name="list_folder", arguments={"path": "~/Documents"}),
                ),
            ),
            Message(role="tool", content="first", tool_name="list_folder"),
            Message(role="tool", content="second", tool_name="list_folder"),
        ]
    )
    await brain.aclose()

    messages = seen[0]["messages"]
    call_ids = [call["id"] for call in messages[1]["tool_calls"]]
    assert [message["tool_call_id"] for message in messages[2:]] == call_ids


async def test_openai_rejects_unreadable_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "list_folder", "arguments": "{not json"},
                                }
                            ]
                        }
                    }
                ]
            },
        )

    brain = _brain(OpenAIBrain, handler)
    with pytest.raises(BrainResponseError):
        await brain.generate([Message(role="user", content="hi")])
    await brain.aclose()


# -- Anthropic shape --------------------------------------------------------


async def test_anthropic_hoists_system_and_reads_tool_use() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "search_files", "input": {"path": "~/Documents"}},
                ],
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        )

    brain = _brain(AnthropicBrain, handler)
    response = await brain.generate(
        [Message(role="system", content="you are ClaireAI"), Message(role="user", content="find pdfs")],
        tools=[{"name": "search_files", "description": "search", "parameters": {"type": "object"}}],
        temperature=2.0,
    )
    await brain.aclose()

    assert response.content == "Checking."
    assert response.tool_calls[0].name == "search_files"
    assert response.tool_calls[0].arguments == {"path": "~/Documents"}
    payload = seen[0]
    assert payload["system"] == "you are ClaireAI"
    assert all(message["role"] != "system" for message in payload["messages"])
    assert payload["temperature"] == 1.0  # clamped to the API's range
    assert payload["tools"][0]["input_schema"] == {"type": "object"}


async def test_anthropic_merges_consecutive_tool_results_into_one_user_turn() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    brain = _brain(AnthropicBrain, handler)
    await brain.generate(
        [
            Message(role="user", content="two things"),
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(name="list_folder", arguments={"path": "~/Downloads"}),
                    ToolCall(name="list_folder", arguments={"path": "~/Documents"}),
                ),
            ),
            Message(role="tool", content="first", tool_name="list_folder"),
            Message(role="tool", content="second", tool_name="list_folder"),
        ]
    )
    await brain.aclose()

    messages = seen[0]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    results = messages[2]["content"]
    assert [block["type"] for block in results] == ["tool_result", "tool_result"]
    call_ids = [block["id"] for block in messages[1]["content"]]
    assert [block["tool_use_id"] for block in results] == call_ids


# -- Ollama shape -----------------------------------------------------------


async def test_ollama_reads_object_arguments_and_reports_tools() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_file_info", "arguments": {"path": "~/a.txt"}}}
                    ],
                },
                "prompt_eval_count": 30,
                "eval_count": 8,
            },
        )

    brain = _brain(OllamaBrain, handler, api_key=None)
    response = await brain.generate(
        [Message(role="user", content="size of a.txt")],
        tools=[{"name": "get_file_info", "description": "info", "parameters": {"type": "object"}}],
    )
    await brain.aclose()

    assert response.tool_calls[0].arguments == {"path": "~/a.txt"}
    payload = seen[0]
    assert payload["stream"] is False
    assert payload["tools"][0]["function"]["name"] == "get_file_info"


async def test_ollama_health_explains_a_missing_pull() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/tags")
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})

    brain = _brain(OllamaBrain, handler, api_key=None)
    health = await brain.health()
    await brain.aclose()

    assert health.available is True
    assert health.model_present is False
    assert "ollama pull" in health.detail


# -- failure handling -------------------------------------------------------


async def test_auth_failure_names_the_provider_and_never_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    brain = _brain(OpenAIBrain, handler)
    with pytest.raises(BrainResponseError) as raised:
        await brain.generate([Message(role="user", content="hi")])
    await brain.aclose()

    message = str(raised.value)
    assert "sk-secret-value" not in message
    assert "OpenAI" in message
    assert "CLAIREAI_BRAIN_API_KEY" in message


async def test_an_unreachable_endpoint_is_unavailable_not_a_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    brain = _brain(AnthropicBrain, handler)
    with pytest.raises(BrainUnavailable):
        await brain.generate([Message(role="user", content="hi")])
    await brain.aclose()
