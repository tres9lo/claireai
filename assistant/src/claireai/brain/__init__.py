"""Brain package: the provider-neutral interface and its adapters."""

from claireai.brain.base import (
    Brain,
    BrainError,
    BrainHealth,
    BrainResponse,
    BrainResponseError,
    BrainStopped,
    BrainUnavailable,
    Message,
    ToolCall,
)
from claireai.brain.anthropic import AnthropicBrain
from claireai.brain.factory import PROVIDERS, PROVIDER_DEFAULT_BASE_URLS, build_brain
from claireai.brain.ollama import OllamaBrain
from claireai.brain.openai import OpenAIBrain
from claireai.brain.scratch import ScratchBrain

__all__ = [
    "AnthropicBrain",
    "Brain",
    "BrainError",
    "BrainHealth",
    "BrainResponse",
    "BrainResponseError",
    "BrainStopped",
    "BrainUnavailable",
    "Message",
    "OllamaBrain",
    "OpenAIBrain",
    "PROVIDERS",
    "PROVIDER_DEFAULT_BASE_URLS",
    "ScratchBrain",
    "ToolCall",
    "build_brain",
]
