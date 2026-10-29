"""Ollama adapter for the local model runtime.

Uses Ollama's local HTTP API (POST /api/chat, GET /api/tags) verified against
https://docs.ollama.com/api/chat on 2026-10-29. No hosted provider is involved
and no API key is read.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import httpx

from claireai.brain.base import (
    BrainHealth,
    BrainResponse,
    BrainResponseError,
    BrainUnavailable,
    Message,
)


class OllamaBrain:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:1.5b",
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> BrainHealth:
        try:
            response = await self._client.get("/api/tags")
        except httpx.HTTPError as exc:
            return BrainHealth(
                available=False,
                model_present=False,
                detail=f"cannot reach Ollama at {self.base_url}: {exc.__class__.__name__}",
            )
        if response.status_code != 200:
            return BrainHealth(
                available=False,
                model_present=False,
                detail=f"Ollama at {self.base_url} answered HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError:
            return BrainHealth(available=True, model_present=False, detail="Ollama returned non-JSON /api/tags")

        models = tuple(
            str(entry.get("name") or entry.get("model") or "")
            for entry in payload.get("models", [])
            if isinstance(entry, Mapping)
        )
        return BrainHealth(
            available=True,
            model_present=self._model_installed(models),
            models=models,
            detail=f"{len(models)} model(s) installed",
        )

    def _model_installed(self, models: Sequence[str]) -> bool:
        wanted = self.model
        for name in models:
            # Ollama reports tagged names; `qwen2.5:1.5b` and `qwen2.5:1.5b`
            # match directly, while a bare name implies the `:latest` tag.
            if name == wanted or name == f"{wanted}:latest" or wanted == f"{name}:latest":
                return True
        return False

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> BrainResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_wire() for message in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)

        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise BrainUnavailable(
                f"Ollama did not answer within the configured timeout ({self.base_url}). "
                "The model may still be loading; retry, or raise CLAIREAI_OLLAMA_TIMEOUT_SECONDS."
            ) from exc
        except httpx.HTTPError as exc:
            raise BrainUnavailable(
                f"Cannot reach Ollama at {self.base_url} ({exc.__class__.__name__}). "
                "Start it with 'ollama serve' and confirm 'ollama list' shows the model."
            ) from exc

        if response.status_code == 404:
            raise BrainResponseError(
                f"Ollama does not have model {self.model!r}. Download it locally first: "
                f"ollama pull {self.model}"
            )
        if response.status_code >= 400:
            body = response.text[:200]
            raise BrainResponseError(f"Ollama returned HTTP {response.status_code}: {body}")

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise BrainResponseError("Ollama returned an unexpected /api/chat response shape") from exc

        return BrainResponse(
            content=str(content),
            model=str(data.get("model", self.model)),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
