"""Ollama adapter (native ``/api/chat``).

Ollama runs models on the owner's machine, so this connector keeps the "my data
stays here" property while letting a larger local model answer. It is still an
opt-in: the scratch checkpoint remains the default.

The native API differs from the OpenAI shape in that tool arguments come back as
an object rather than a JSON string, and a tool message carries ``tool_name``
instead of an id, which maps directly onto ``Message``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from claireai.brain.base import BrainError, BrainHealth, BrainResponseError, ToolCall
from claireai.brain.remote import RemoteBrain, messages_for_http, parse_arguments


class OllamaBrain(RemoteBrain):
    name = "ollama"
    provider = "Ollama"
    default_base_url = "http://127.0.0.1:11434/api"
    requires_api_key = False

    def headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    async def health(self) -> BrainHealth:
        try:
            payload = await self._request("GET", "/tags")
        except BrainError as exc:
            return BrainHealth(False, False, detail=str(exc))
        models = tuple(
            str(item.get("name") or item.get("model"))
            for item in payload.get("models") or []
            if isinstance(item, Mapping) and (item.get("name") or item.get("model"))
        )
        present = self.model in models or f"{self.model}:latest" in models
        detail = (
            f"Ollama has {self.model}"
            if present
            else f"Ollama is running but {self.model} is not pulled (try: ollama pull {self.model})"
        )
        return BrainHealth(True, present, models, detail)

    async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001, ANN201
        options: dict[str, Any] = {"num_predict": self.max_tokens}
        if temperature is not None:
            options["temperature"] = max(0.0, float(temperature))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _wire_messages(messages),
            "stream": False,
            "options": options,
        }
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": str(function.get("name")),
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for schema in tools or ()
            if (function := self._function(schema)).get("name")
        ]
        if schemas:
            payload["tools"] = schemas

        data = await self._request("POST", "/chat", payload=payload)
        message = data.get("message") or {}
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            function = (item or {}).get("function") or {}
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            calls.append(
                ToolCall(
                    name=name,
                    arguments=parse_arguments(function.get("arguments"), provider=self.provider),
                    index=index,
                )
            )
        if not calls and not str(message.get("content") or "").strip():
            raise BrainResponseError("Ollama returned an empty message")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        return self._response(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            model=str(data.get("model") or self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _wire_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages_for_http(messages):
        if message.role == "assistant" and message.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {"function": {"name": call.name, "arguments": call.arguments}}
                        for call in message.tool_calls
                    ],
                }
            )
        elif message.role == "tool":
            entry: dict[str, Any] = {"role": "tool", "content": message.content}
            if message.tool_name:
                entry["tool_name"] = message.tool_name
            out.append(entry)
        else:
            out.append({"role": message.role, "content": message.content})
    return out
