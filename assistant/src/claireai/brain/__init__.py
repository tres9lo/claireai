"""Brain package: the provider-neutral interface and its adapters."""

from claireai.brain.base import (
    Brain,
    BrainError,
    BrainHealth,
    BrainResponse,
    BrainResponseError,
    BrainUnavailable,
    Message,
)
from claireai.brain.ollama import OllamaBrain

__all__ = [
    "Brain",
    "BrainError",
    "BrainHealth",
    "BrainResponse",
    "BrainResponseError",
    "BrainUnavailable",
    "Message",
    "OllamaBrain",
]
