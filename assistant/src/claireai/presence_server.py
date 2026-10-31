"""Local event server for the presence renderer (Month 1, Week 4 groundwork).

Deliberately dependency-free: Python's standard library serves the HUD and the
event stream, so the renderer can be run and inspected before the desktop shell
(Rust + npm) is installed. When the Tauri window exists it will consume the same
contract over the same loopback API.

Security posture:

* binds to loopback only unless the caller explicitly says otherwise;
* serves one static HTML file and JSON state, nothing from disk by request path;
* no CORS headers, no cookies, no auth bypass — it is a local view of local state;
* it holds no capability: the stream it serves is read-only.
"""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from claireai.presence import STATES, PresenceStream, sse_frame

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
POLL_SECONDS = 0.2


def _handler_factory(stream: PresenceStream, html: bytes, page_title: str) -> type[BaseHTTPRequestHandler]:
    class PresenceHandler(BaseHTTPRequestHandler):
        server_version = "ClaireAIPresence/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Keep the terminal readable; the preview is a development tool.
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", html)
            elif path == "/state":
                payload = json.dumps(stream.snapshot().as_dict(), sort_keys=True).encode()
                self._send(HTTPStatus.OK, "application/json", payload)
            elif path == "/events":
                self._stream()
            elif path == "/healthz":
                self._send(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok\n")
            else:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 1000\n\n")
                self.wfile.flush()
                last = -1
                while True:
                    snapshot = stream.snapshot()
                    if snapshot.seq != last:
                        last = snapshot.seq
                        self.wfile.write(sse_frame(snapshot).encode("utf-8"))
                        self.wfile.flush()
                    time.sleep(POLL_SECONDS)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    PresenceHandler.page_title = page_title  # type: ignore[attr-defined]
    return PresenceHandler


def build_server(
    stream: PresenceStream,
    *,
    html_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create (but do not start) the preview server."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"the presence preview binds to loopback only, refusing host {host!r}; "
            "a LAN-exposed renderer needs the paired API and a threat review"
        )
    html = html_path.read_bytes()
    handler = _handler_factory(stream, html, html_path.name)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def demo_loop(stream: PresenceStream, *, stop: threading.Event, delay: float = 1.5) -> None:
    """Cycle states so the HUD can be seen without running the model."""
    script = [
        ("idle", {}, {"microphone": False, "camera": False, "screen": False, "input": False}),
        ("listening", {"label": "push-to-talk"}, {"microphone": True}),
        ("thinking", {"label": "model generating"}, {"microphone": False}),
        ("tool_running", {"tool": "find_largest_files"}, {}),
        ("awaiting_approval", {"tool": "move_file", "ref": "demo"}, {}),
        ("speaking", {"label": "reading the answer"}, {}),
        ("idle", {"label": "ready"}, {"camera": False, "screen": False, "input": False}),
    ]
    index = 0
    while not stop.is_set():
        state, detail, active = script[index % len(script)]
        stream.emit(state, task_id="demo-task", detail=detail, active=active)
        index += 1
        stop.wait(delay)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the ClaireAI presence HUD.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="cycle fake states for a visual check")
    parser.add_argument(
        "--html",
        default=str(Path(__file__).resolve().parents[3] / "desktop" / "presence" / "index.html"),
    )
    args = parser.parse_args(argv)

    stream = PresenceStream()
    stop = threading.Event()
    if args.demo:
        threading.Thread(target=demo_loop, args=(stream,), kwargs={"stop": stop}, daemon=True).start()

    server = build_server(stream, html_path=Path(args.html), host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"presence preview: http://{host}:{port}/  ({len(STATES)} states)")
    if args.demo:
        print("demo mode: fake state changes every 1.5 s")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
