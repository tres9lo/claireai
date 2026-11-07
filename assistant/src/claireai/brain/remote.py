"""Shared plumbing for the connector brains.

The from-scratch model is the default brain and makes no network request. These
adapters exist for the case the owner has decided the local model is not enough
for a task: they are opt-in, they send the conversation (and the tool schemas)
to an endpoint the owner configured, and the audit row names the provider that
answered. Nothing here runs unless ``CLAIREAI_BRAIN_PROVIDER`` selects it.

What this module guarantees, in code rather than in a document:

- the endpoint is validated by :func:`claireai.config.endpoint.check_endpoint`,
  so a key is never sent over plain http to a non-loopback host;
- an authentication or protocol failure surfaces as a ``BrainError`` with the
  endpoint and status, never the key;
- one ``httpx.AsyncClient`` is shared per brain and closed by ``aclose()``, so a
  long-running session does not open a connection per turn.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import httpx

from claireai.brain.base import (
    BrainResponseError,
    BrainUnavailable,
    Message,
    ToolCall,
)
from claireai.config.endpoint import check_endpoint


class RemoteBrain:
    """Common HTTP behaviour for every connector."""

    name = "remote"
    provider = "remote"
    default_base_url = ""
    requires_api_key = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
    ) -> None:
        if not str(model or "").strip():
            raise ValueError(f"{self.provider}: a model name is required")
        if self.requires_api_key and not api_key:
            raise ValueError(f"{self.provider}: an API key is required")
        self.model = str(model).strip()
        self.base_url = check_endpoint(str(base_url or self.default_base_url))
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self._http: httpx.AsyncClient | None = None

    # -- transport ---------------------------------------------------------
    def headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = await self._client().request(
                method, url, json=payload, params=params, headers=self.headers()
            )
        except httpx.TimeoutException as exc:
            raise BrainUnavailable(
                f"{self.provider} did not answer within {self.timeout:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise BrainUnavailable(
                f"{self.provider} is unreachable at {self.base_url} ({type(exc).__name__})"
            ) from exc

        if response.status_code in {401, 403}:
            raise BrainResponseError(
                f"{self.provider} rejected the credentials (HTTP {response.status_code}); "
                "check CLAIREAI_BRAIN_API_KEY"
            )
        if response.status_code == 429:
            raise BrainUnavailable(f"{self.provider} is rate limiting this client (HTTP 429)")
        if response.status_code >= 400:
            excerpt = " ".join(response.text.split())[:200]
            raise BrainResponseError(f"{self.provider} returned HTTP {response.status_code}: {excerpt}")
        try:
            return response.json()
        except ValueError as exc:
            raise BrainResponseError(f"{self.provider} returned a response that is not JSON") from exc

    # -- helpers -----------------------------------------------------------
    def _response(
        self,
        *,
        content: str,
        tool_calls: list[ToolCall],
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> Any:
        from claireai.brain.base import BrainResponse

        return BrainResponse(
            content=content,
            model=f"{self.name}:{model or self.model}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tuple(tool_calls),
        )

    @staticmethod
    def _function(schema: Mapping[str, Any]) -> Mapping[str, Any]:
        """The function object inside a tool schema, whatever the wrapper."""
        inner = schema.get("function")
        return inner if isinstance(inner, Mapping) else schema


def messages_for_http(messages: Any) -> list[Message]:
    """Coerce a message sequence to ``Message`` (adapters accept both)."""
    return [
        message if isinstance(message, Message) else Message(
            role=getattr(message, "role", "user"),
            content=getattr(message, "content", "") or "",
            tool_calls=tuple(getattr(message, "tool_calls", ()) or ()),
            tool_name=getattr(message, "tool_name", None),
        )
        for message in messages
    ]


def parse_arguments(value: Any, *, provider: str) -> dict[str, Any]:
    """Tool arguments arrive as an object, or as a JSON string of one."""
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            raise BrainResponseError(f"{provider} sent tool arguments that are not valid JSON") from exc
        if isinstance(decoded, Mapping):
            return dict(decoded)
    raise BrainResponseError(f"{provider} sent tool arguments that are not a JSON object")
