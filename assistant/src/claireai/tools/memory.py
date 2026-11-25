"""Tools for the local semantic memory index."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from claireai.memory import EmbeddingError, MAX_RESULTS, render_hits
from claireai.tools.base import ToolContext, ToolError, ToolResult


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RememberNoteArgs(_Args):
    text: str = Field(
        min_length=1,
        max_length=8_000,
        description="The fact, preference or note to keep in local memory.",
    )


class SearchMemoryArgs(_Args):
    query: str = Field(
        min_length=1,
        max_length=2_000,
        description="What to look for in the owner's stored memory.",
    )
    limit: int = Field(default=5, ge=1, le=MAX_RESULTS)


def remember_note(context: ToolContext, arguments: RememberNoteArgs) -> ToolResult:
    memory = getattr(context, "memory", None)
    if memory is None:
        raise ToolError("local memory is disabled")
    try:
        entry_id = memory.add(arguments.text, kind="note")
    except (EmbeddingError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    return ToolResult(
        text="Saved to local memory.",
        detail={"id": entry_id, "characters": len(arguments.text), "text": "[redacted]"},
    )


def search_memory(context: ToolContext, arguments: SearchMemoryArgs) -> ToolResult:
    memory = getattr(context, "memory", None)
    if memory is None:
        raise ToolError("local memory is disabled")
    try:
        hits = memory.search(arguments.query, limit=arguments.limit)
    except (EmbeddingError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    return ToolResult(
        text=render_hits(arguments.query, hits),
        detail={
            "query": arguments.query,
            "results": [
                {"id": hit.id, "score": round(hit.score, 4), "kind": hit.kind}
                for hit in hits
            ],
            "model": memory.embedder.name,
        },
    )
