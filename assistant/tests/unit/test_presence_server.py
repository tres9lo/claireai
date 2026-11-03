"""The preview server: loopback only, read-only, and honest over HTTP."""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from claireai.presence import PresenceStream
from claireai.presence_server import build_server
from claireai.improvements import ImprovementLedger
from claireai.brain import BrainHealth, BrainResponse
from claireai.storage import ConversationRepository, Database, TaskRepository, transaction

HTML = Path(__file__).resolve().parents[3] / "desktop" / "presence" / "index.html"


@pytest.fixture()
def server():
    stream = PresenceStream()
    instance = build_server(stream, html_path=HTML, port=0)
    thread = threading.Thread(target=instance.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield stream, instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def _get(server, path: str, timeout: float = 3.0):  # noqa: ANN001, ANN202
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read()
    headers = dict(response.getheaders())
    connection.close()
    return response.status, headers, body


def test_the_hud_is_served(server) -> None:  # noqa: ANN001
    _, instance = server
    status, headers, body = _get(instance, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<canvas" in body


def test_the_state_endpoint_matches_the_stream(server) -> None:  # noqa: ANN001
    stream, instance = server
    status, headers, body = _get(instance, "/state")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["state"] == "idle"

    stream.emit("thinking", task_id="t1")
    payload = json.loads(_get(instance, "/state")[2])
    assert payload["state"] == "thinking"
    assert payload["taskId"] == "t1"
    assert payload["seq"] == 1


def test_the_event_stream_sends_the_current_state(server) -> None:  # noqa: ANN001
    stream, instance = server
    stream.emit("awaiting_approval", task_id="t2", detail={"tool": "move_file"})

    host, port = instance.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=3)
    connection.request("GET", "/events")
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"

    # The stream flushes `retry:` before the first event, so a single read can
    # legitimately return only that line. Read until a data frame arrives.
    buffer = ""
    deadline = time.time() + 3
    while "data: " not in buffer and time.time() < deadline:
        chunk = response.read1(4096).decode("utf-8")
        if not chunk:
            break
        buffer += chunk

    assert "event: presence" in buffer
    payload = json.loads(buffer.split("data: ", 1)[1].split("\n", 1)[0])
    assert payload["state"] == "awaiting_approval"
    assert payload["detail"]["tool"] == "move_file"
    connection.close()


def test_unknown_paths_are_404(server) -> None:  # noqa: ANN001
    _, instance = server
    assert _get(instance, "/etc/passwd")[0] == 404


def test_a_health_check_exists(server) -> None:  # noqa: ANN001
    _, instance = server
    status, _, body = _get(instance, "/healthz")
    assert status == 200
    assert body.strip() == b"ok"


def test_the_server_refuses_a_non_loopback_bind() -> None:
    with pytest.raises(ValueError, match="loopback only"):
        build_server(PresenceStream(), html_path=HTML, host="0.0.0.0", port=0)


# --- the read-only /today page ------------------------------------------------


@pytest.fixture()
def data_server(tmp_path: Path):
    from claireai.audit import AuditLog
    from claireai.storage import Database, TaskRepository, transaction
    from claireai.summary import build, save

    database = Database(tmp_path / "db" / "claireai.sqlite3")
    database.migrate()
    conn = database.connect()
    tasks = TaskRepository(conn)
    with transaction(conn):
        task = tasks.create(request="List the biggest files in my Downloads folder", kind="agent")
        tasks.update_status(task.id, "running")
        tasks.update_status(task.id, "succeeded")
    with transaction(conn):
        AuditLog(conn).record(task_id=task.id, tool="find_largest_files", action="tool.call", status="done", detail={})
        save(conn, build(conn), data_dir=tmp_path)
    conn.close()

    stream = PresenceStream()
    instance = build_server(
        stream,
        html_path=HTML,
        port=0,
        database_path=database.path,
        control_path=tmp_path / "data" / "runtime-control.json",
    )
    thread = threading.Thread(target=instance.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield database.path, instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def test_today_renders_the_summary_and_the_log(data_server) -> None:  # noqa: ANN001
    _, instance = data_server
    status, headers, body = _get(instance, "/today")
    text = body.decode("utf-8")

    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Content-Security-Policy"].startswith("default-src 'none'")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "biggest files" in text
    assert "find_largest_files" in text


def test_today_cannot_write_to_the_database(data_server) -> None:  # noqa: ANN001
    path, instance = data_server
    before = path.stat().st_mtime_ns
    _get(instance, "/today")
    _get(instance, "/today")
    assert path.stat().st_mtime_ns == before


def test_local_ui_saves_directives_preferences_and_conversation_history(data_server) -> None:  # noqa: ANN001
    path, instance = data_server
    host, port = instance.server_address[:2]
    origin = f"http://{host}:{port}"
    connection = http.client.HTTPConnection(host, port, timeout=3)
    headers = {"Origin": origin, "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"}
    try:
        connection.request(
            "POST", "/api/directives",
            body=json.dumps({"title": "Formatting", "content": "Use clear headings."}),
            headers=headers,
        )
        response = connection.getresponse()
        directives = json.loads(response.read())
        assert response.status == 200 and directives[0]["content"] == "Use clear headings."

        connection.request(
            "POST", "/api/preferences",
            body=json.dumps({"owner_name": "Ange", "response_style": "detailed", "voice_pitch": 66}),
            headers=headers,
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 200

        database = Database(path)
        conn = database.connect()
        try:
            with transaction(conn):
                conversation_id = ConversationRepository(conn).create()
                task = TaskRepository(conn).create(request="redacted request", kind="chat")
                ConversationRepository(conn).add_message(
                    conversation_id, role="user", content="private local question", task_id=task.id
                )
                ConversationRepository(conn).add_message(
                    conversation_id, role="assistant", content="private local answer", task_id=task.id
                )
        finally:
            conn.close()

        connection.request("GET", f"/api/conversations?id={conversation_id}")
        response = connection.getresponse()
        messages = json.loads(response.read())
        assert response.status == 200
        assert [row["content"] for row in messages] == ["private local question", "private local answer"]

        connection.request(
            "GET", "/api/preferences",
            headers={"Origin": "http://attacker.invalid", "Sec-Fetch-Site": "cross-site"},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 403
    finally:
        connection.close()


def test_web_chat_uses_local_brain_and_persists_both_sides(data_server, monkeypatch) -> None:  # noqa: ANN001
    path, instance = data_server
    host, port = instance.server_address[:2]
    origin = f"http://{host}:{port}"
    observed = {}

    class StubBrain:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.max_output_tokens = kwargs["max_output_tokens"]

        async def health(self):
            return BrainHealth(True, True, detail="test")

        async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001
            observed["messages"] = list(messages)
            observed["temperature"] = temperature
            return BrainResponse(content="local answer", model="test-scratch")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("claireai.presence_server.ScratchBrain", StubBrain)
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request(
            "POST", "/api/chat", body=json.dumps({"message": "hello"}),
            headers={"Origin": origin, "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert result["content"] == "local answer"
        assert result["conversation_id"]
        assert observed["messages"][-1].content == "hello"

        connection.request("GET", f"/api/conversations?id={result['conversation_id']}")
        response = connection.getresponse()
        messages = json.loads(response.read())
        assert [row["role"] for row in messages] == ["user", "assistant"]
        assert [row["content"] for row in messages] == ["hello", "local answer"]
    finally:
        connection.close()


def test_there_is_no_way_to_post_an_approval(data_server) -> None:  # noqa: ANN001
    _, instance = data_server
    host, port = instance.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=3)
    try:
        for path in ("/approve", "/today", "/run"):
            connection.request("POST", path, body=b"approve=1")
            response = connection.getresponse()
            response.read()
            # Unrecognized POST routes have no side effect.
            assert response.status == 404
    finally:
        connection.close()


def test_control_posts_require_same_origin_and_json(tmp_path: Path) -> None:
    control_path = tmp_path / "runtime-control.json"
    instance = build_server(
        PresenceStream(), html_path=HTML, port=0, control_path=control_path
    )
    thread = threading.Thread(target=instance.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    host, port = instance.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=3)
    origin = f"http://{host}:{port}"
    try:
        connection.request(
            "POST",
            "/control",
            body=json.dumps({"browser": True}),
            headers={"Origin": origin, "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"},
        )
        response = connection.getresponse()
        state = json.loads(response.read())
        assert response.status == 200
        assert state == {"browser": True, "shutdown": False, "self_improvement": False}

        connection.request(
            "POST",
            "/control",
            body=json.dumps({"shutdown": True}),
            headers={"Origin": "http://attacker.invalid", "Content-Type": "application/json", "Sec-Fetch-Site": "cross-site"},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 403
        assert json.loads(control_path.read_text(encoding="utf-8")) == state

        connection.request(
            "POST", "/control", body=json.dumps({"self_improvement": True}),
            headers={"Origin": origin, "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"},
        )
        response = connection.getresponse()
        state = json.loads(response.read())
        assert response.status == 200 and state["self_improvement"] is True

        # Same-site origin is still rejected when the request uses a rebinding
        # hostname instead of the loopback Host required by this service.
        connection.request(
            "POST",
            "/control",
            body=json.dumps({"shutdown": True}),
            headers={"Host": "attacker.invalid", "Origin": "http://attacker.invalid", "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 403
        assert json.loads(control_path.read_text(encoding="utf-8")) == state
    finally:
        connection.close()
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def test_improvement_log_can_be_managed_only_after_passing_fixed_evaluation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    candidate_path = data_dir / "models" / "candidates" / "candidate.pt"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_bytes(b"candidate")
    ledger = ImprovementLedger(data_dir)
    candidate_id = ledger.record_candidate(checkpoint=candidate_path, parent=candidate_path, manifest={})
    ledger.record_evaluation(candidate_id, {"all_stable": True, "cases": 1}, eligible=False)
    control_path = data_dir / "runtime-control.json"
    instance = build_server(PresenceStream(), html_path=HTML, port=0, control_path=control_path)
    thread = threading.Thread(target=instance.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    host, port = instance.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=3)
    origin = f"http://{host}:{port}"
    try:
        connection.request("GET", "/improvements")
        response = connection.getresponse()
        items = json.loads(response.read())
        assert response.status == 200 and items[0]["status"] == "evaluation_failed"

        body = json.dumps({"id": candidate_id, "action": "approve"})
        connection.request(
            "POST", "/improvements", body=body,
            headers={"Origin": origin, "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"},
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        assert ledger.list()[0]["status"] == "evaluation_failed"
    finally:
        connection.close()
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def test_today_without_a_database_explains_itself(server) -> None:  # noqa: ANN001
    _, instance = server
    status, _, body = _get(instance, "/today")
    assert status == 200
    assert "No task log is configured" in body.decode("utf-8")
