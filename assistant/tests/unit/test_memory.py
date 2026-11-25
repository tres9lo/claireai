"""Local semantic memory without Ollama or a network."""

from __future__ import annotations

import sqlite3

import pytest

from claireai.config import Settings
from claireai.memory import (
    HashingEmbedder,
    MemoryIndex,
    OllamaEmbedder,
    embedder_for,
    memory_index,
)
from claireai.permissions import Sandbox
from claireai.tools.base import ToolContext, ToolError
from claireai.tools.memory import RememberNoteArgs, SearchMemoryArgs, remember_note, search_memory


def test_hash_embeddings_are_stable_and_normalised() -> None:
    embedder = HashingEmbedder(64)
    first = embedder.embed(["budget PC and local models"])[0]
    second = embedder.embed(["budget PC and local models"])[0]
    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_memory_search_returns_the_closest_note(conn: sqlite3.Connection) -> None:
    index = MemoryIndex(conn, HashingEmbedder())
    index.add("The owner uses a budget CPU-only PC.")
    index.add("The owner prefers French in the evening.")
    index.add("The owner's calendar is local.")

    hits = index.search("budget CPU machine", limit=2)

    assert hits
    assert "budget" in hits[0].text
    assert hits[0].score > 0


def test_a_source_id_replaces_the_previous_entry(conn: sqlite3.Connection) -> None:
    index = MemoryIndex(conn, HashingEmbedder())
    first = index.add("old turn", kind="turn", source_id="task-1")
    second = index.add("new turn", kind="turn", source_id="task-1")

    assert first != second
    assert index.count() == 1
    assert index.search("new turn")[0].text == "new turn"


def test_memory_can_be_deleted_and_counted(conn: sqlite3.Connection) -> None:
    index = MemoryIndex(conn, HashingEmbedder())
    entry = index.add("temporary")
    assert index.count() == 1
    assert index.delete(entry) is True
    assert index.delete(entry) is False
    assert index.count() == 0


def test_memory_tools_use_the_injected_index(conn: sqlite3.Connection) -> None:
    index = MemoryIndex(conn, HashingEmbedder())
    context = ToolContext(
        sandbox=Sandbox(roots=()),
        memory=index,
    )

    saved = remember_note(context, RememberNoteArgs(text="The owner likes quiet mornings."))
    found = search_memory(context, SearchMemoryArgs(query="quiet mornings"))

    assert "Saved" in saved.text
    assert "quiet mornings" in found.text
    assert "password" not in saved.text


def test_memory_tool_refuses_when_disabled() -> None:
    with pytest.raises(ToolError, match="disabled"):
        search_memory(
            ToolContext(sandbox=Sandbox(roots=())),
            SearchMemoryArgs(query="anything"),
        )


def test_memory_index_respects_the_setting(conn: sqlite3.Connection) -> None:
    settings = Settings(_env_file=None, memory_enabled=False)
    assert memory_index(settings, conn) is None
    enabled = Settings(_env_file=None, memory_enabled=True, embedding_provider="hash")
    assert isinstance(memory_index(enabled, conn), MemoryIndex)


def test_ollama_embedder_parses_a_batch_response() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/embed")
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0], [0.0, 2.0]]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OllamaEmbedder(
        base_url="http://127.0.0.1:11434/api",
        model="nomic-embed-text",
        client=client,
    )
    vectors = embedder.embed(["one", "two"])
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]


def test_ollama_embedder_reports_a_missing_model() -> None:
    import httpx

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(404, json={"error": "model not found"})
        )
    )
    embedder = OllamaEmbedder(
        base_url="http://127.0.0.1:11434/api",
        model="nomic-embed-text",
        client=client,
    )
    with pytest.raises(Exception, match="ollama pull"):
        embedder.embed(["hello"])


def test_hash_embedder_is_selected_explicitly() -> None:
    settings = Settings(_env_file=None, embedding_provider="hash")
    assert isinstance(embedder_for(settings), HashingEmbedder)
