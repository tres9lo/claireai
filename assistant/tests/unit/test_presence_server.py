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
