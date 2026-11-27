"""Choose the brain the configuration asks for.

``ollama`` is the default: a local Ollama daemon serves Qwen3 1.7B, so the
assistant no longer needs a checkpoint trained in this repository. ``scratch``
is retained for the archived research models, and the hosted adapters remain
opt-in. The loopback Ollama endpoint keeps inference on this machine.

Callers build brains through this function instead of instantiating a class, so
adding a provider is one entry here rather than an edit at every call site.
"""

from __future__ import annotations

from typing import Any

from claireai.brain.anthropic import AnthropicBrain
from claireai.brain.base import Brain
from claireai.brain.ollama import OllamaBrain
from claireai.brain.openai import OpenAIBrain
from claireai.brain.scratch import ScratchBrain
from claireai.config.endpoint import DEFAULT_BASE_URLS

PROVIDERS = ("scratch", "ollama", "openai", "anthropic")
PROVIDER_DEFAULT_BASE_URLS = dict(DEFAULT_BASE_URLS)


def build_brain(settings: Any) -> Brain:
    """Return the configured brain. Raises ``ValueError`` on a bad provider."""
    provider = str(getattr(settings, "brain_provider", "ollama") or "ollama")
    if provider == "scratch":
        return ScratchBrain(
            checkpoint=settings.resolved_model_checkpoint,
            max_output_tokens=settings.max_output_tokens,
            control_path=settings.control_state_path,
        )
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown brain provider {provider!r}; choose one of {', '.join(PROVIDERS)}"
        )

    model = str(getattr(settings, "brain_model", None) or "").strip()
    if not model:
        raise ValueError(f"CLAIREAI_BRAIN_MODEL is required when CLAIREAI_BRAIN_PROVIDER={provider}")

    brain_class = {"ollama": OllamaBrain, "openai": OpenAIBrain, "anthropic": AnthropicBrain}[provider]
    base_url = getattr(settings, "resolved_brain_base_url", None) or PROVIDER_DEFAULT_BASE_URLS[provider]
    options: dict[str, Any] = {}
    if provider == "ollama":
        options["coding_model"] = str(getattr(settings, "brain_coding_model", "") or "") or None
        options["think"] = bool(getattr(settings, "brain_think", False))
    return brain_class(
        model=model,
        base_url=base_url,
        api_key=getattr(settings, "brain_api_key_value", None),
        timeout=getattr(settings, "brain_timeout_seconds", 180.0),
        max_tokens=getattr(settings, "brain_max_tokens", 1024),
        **options,
    )
