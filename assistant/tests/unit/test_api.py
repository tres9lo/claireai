"""The local API service, exercised through ASGI; no network listener is opened."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from claireai.api.app import EventBus, create_app  # noqa: E402
from claireai.approval import ApprovalRequest  # noqa: E402
from claireai.approval_file import issue as issue_approval  # noqa: E402
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


def test_task_lifecycle_runs_through_the_existing_agent(tmp_path, monkeypatch) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    app = create_app(settings, token="test-token")
    stub = StubBrain()
    monkeypatch.setattr("claireai.api.app.build_brain", lambda _settings: stub)

    with TestClient(app) as client:
        created = client.post(
            "/v1/tasks",
            headers={"Authorization": "Bearer test-token"},
            json={"message": "hello", "use_tools": False},
        )
        assert created.status_code == 200
        task_id = created.json()["id"]
        for _ in range(100):
            response = client.get(
                f"/v1/tasks/{task_id}",
                headers={"Authorization": "Bearer test-token"},
            )
            if response.json()["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert response.json()["status"] == "succeeded"
        assert response.json()["audit"]
        listed = client.get(
            "/v1/tasks", headers={"Authorization": "Bearer test-token"}
        )
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == task_id


def test_pairing_creates_a_revocable_device_token(tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    app = create_app(settings, token="admin-token")

    with TestClient(app) as client:
        paired = client.post(
            "/v1/devices/pair",
            headers={"Authorization": "Bearer admin-token"},
            json={"name": "phone"},
        )
        assert paired.status_code == 200
        code = paired.json()["code"]
        claimed = client.post("/v1/devices/claim", json={"code": code})
        assert claimed.status_code == 200
        token = claimed.json()["token"]
        assert client.get(
            "/v1/status", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 200
        devices = client.get(
            "/v1/devices", headers={"Authorization": "Bearer admin-token"}
        ).json()
        assert devices[0]["name"] == "phone"
        device_id = devices[0]["id"]
        revoked = client.delete(
            f"/v1/devices/{device_id}",
            headers={"Authorization": "Bearer admin-token"},
        )
        assert revoked.status_code == 200
        assert client.get(
            "/v1/status", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 401


def test_signed_approvals_can_be_listed_and_denied_but_not_approved(tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    settings.ensure_directories()
    request = ApprovalRequest(
        ref="approval-1",
        task_id="task-1",
        tool="create_folder",
        risk="needs_approval",
        arguments={"path": "~/Documents/reports"},
        target="~/Documents/reports",
    )
    issue_approval(request, data_dir=settings.resolved_data_dir)
    app = create_app(settings, token="admin-token")

    with TestClient(app) as client:
        listed = client.get(
            "/v1/approvals", headers={"Authorization": "Bearer admin-token"}
        )
        assert listed.status_code == 200
        assert listed.json()[0]["tool"] == "create_folder"
        assert client.post(
            f"/v1/approvals/{listed.json()[0]['ref']}/approve",
            headers={"Authorization": "Bearer admin-token"},
        ).status_code == 403
        denied = client.post(
            f"/v1/approvals/{listed.json()[0]['ref']}/deny",
            headers={"Authorization": "Bearer admin-token"},
        )
        assert denied.status_code == 200
        assert client.get(
            "/v1/approvals", headers={"Authorization": "Bearer admin-token"}
        ).json() == []


def test_settings_audit_summary_and_memory_routes(tmp_path) -> None:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", embedding_provider="hash"
    )
    app = create_app(settings, token="admin-token")

    with TestClient(app) as client:
        updated = client.put(
            "/v1/settings",
            headers={"Authorization": "Bearer admin-token"},
            json={"memory_auto_embed": True, "voice_rate": 180},
        )
        assert updated.status_code == 200
        assert updated.json()["memory_auto_embed"] is True
        assert client.put(
            "/v1/settings",
            headers={"Authorization": "Bearer admin-token"},
            json={"api_token": "secret"},
        ).status_code == 422
        summary = client.get(
            "/v1/summary", headers={"Authorization": "Bearer admin-token"}
        )
        assert summary.status_code == 200
        assert "Done" in summary.json()["content"]
        audit = client.get(
            "/v1/audit", headers={"Authorization": "Bearer admin-token"}
        )
        assert audit.status_code == 200
        memory = client.get(
            "/v1/memory", headers={"Authorization": "Bearer admin-token"}
        )
        assert memory.status_code == 200
        assert memory.json()["embedder"] == "hash:384"


def test_memory_entries_can_be_listed_and_deleted(tmp_path) -> None:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", embedding_provider="hash"
    )
    db = Database(settings.database_path)
    db.migrate()
    conn = db.connect()
    try:
        entry = MemoryIndex(conn, HashingEmbedder()).add("delete me locally")
    finally:
        conn.close()
    app = create_app(settings, token="admin-token")

    with TestClient(app) as client:
        listed = client.get(
            "/v1/memory", headers={"Authorization": "Bearer admin-token"}
        )
        assert listed.status_code == 200
        assert listed.json()["entries"][0]["id"] == entry
        deleted = client.delete(
            f"/v1/memory/{entry}", headers={"Authorization": "Bearer admin-token"}
        )
        assert deleted.status_code == 200
        assert client.get(
            "/v1/memory", headers={"Authorization": "Bearer admin-token"}
        ).json()["count"] == 0


async def test_event_bus_emits_ordered_sse_frames() -> None:
    bus = EventBus()
    stream = bus.stream()
    assert "event: ready" in await anext(stream)
    bus.publish("task.completed", task_id="task-1")
    frame = await anext(stream)
    assert "event: task.completed" in frame
    assert '"task_id": "task-1"' in frame
    await stream.aclose()
