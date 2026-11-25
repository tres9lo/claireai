"""The local API service, exercised through ASGI; no network listener is opened."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from claireai.api.app import create_app  # noqa: E402
from claireai.brain import BrainResponse  # noqa: E402
from claireai.config import Settings  # noqa: E402
from claireai.memory import HashingEmbedder, MemoryIndex  # noqa: E402
from claireai.storage import Database  # noqa: E402


class StubBrain:
    name = "stub"
    max_output_tokens = 128

    def __init__(self) -> None:
        self.calls = 0

    async def health(self):  # noqa: ANN201
        from claireai.brain import BrainHealth

        return BrainHealth(available=True, model_present=True, detail="stub")

    async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001
        self.calls += 1
        return BrainResponse(content="Hello from the local API.", model="stub")

    async def aclose(self) -> None:
        return None


def test_health_is_public_and_every_other_route_needs_a_token(tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    app = create_app(settings, token="test-token")

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/status").status_code == 401
        assert (
            client.get("/v1/status", headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        status = client.get(
            "/v1/status", headers={"Authorization": "Bearer test-token"}
        )
        assert status.status_code == 200
        assert status.json()["provider"] == "ollama"


def test_memory_search_returns_local_hits(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedding_provider="hash",
        memory_enabled=True,
    )
    db = Database(settings.database_path)
    db.migrate()
    conn = db.connect()
    try:
        MemoryIndex(conn, HashingEmbedder()).add("The owner works on a budget PC.")
    finally:
        conn.close()
    app = create_app(settings, token="test-token")

    with TestClient(app) as client:
        response = client.post(
            "/v1/memory/search",
            headers={"Authorization": "Bearer test-token"},
            json={"query": "budget PC", "limit": 3},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert "budget PC" in body["text"]


def test_chat_uses_the_runtime_without_tools(tmp_path, monkeypatch) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    app = create_app(settings, token="test-token")
    stub = StubBrain()
    monkeypatch.setattr("claireai.api.app.build_brain", lambda _settings: stub)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer test-token"},
            json={"message": "Say hello", "use_tools": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "Hello from the local API."
    assert body["model"] == "stub"
    assert stub.calls == 1


def test_a_missing_token_file_is_created_without_exposing_it(tmp_path) -> None:
    from claireai.api.app import TOKEN_FILE, load_or_create_token

    token = load_or_create_token(tmp_path)
    path = tmp_path / TOKEN_FILE
    assert path.read_text(encoding="utf-8").strip() == token
    assert len(token) >= 32
    assert path.stat().st_mode & 0o077 == 0
