"""Loopback HUD and same-origin local APIs for ClaireAI.

The text chat and transcription endpoints use local models but do not invoke
computer tools. Control APIs are limited to explicit owner settings, while
``/today`` remains read-only and carries a strict CSP.
"""

from __future__ import annotations

import base64
import binascii
import json
import ipaddress
import sqlite3
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from claireai.presence import STATES, PresenceStream, sse_frame
from claireai.runtime_control import RuntimeControl
from claireai.improvements import ImprovementError, ImprovementLedger
from claireai.today import open_read_only, render_today, render_unavailable
from claireai.config import Settings
from claireai.brain import BrainError, BrainStopped, ScratchBrain
from claireai.storage import (
    ConversationRepository,
    Database,
    DirectiveRepository,
    PreferenceRepository,
)
from claireai.voice_input import SpeechRecognitionUnavailable

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
POLL_SECONDS = 0.2
# No scripts at all on the generated page, and no way to submit anything.
TODAY_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"


def _is_loopback_host_header(raw: str) -> bool:
    host = urlparse(f"//{raw}").hostname
    if not host:
        return False
    if host.lower() in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _handler_factory(
    stream: PresenceStream,
    html: bytes,
    page_title: str,
    database_path: Path | None = None,
    control_path: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    voice_state_lock = threading.Lock()
    voice_state_timer: threading.Timer | None = None
    voice_state_generation = 0

    def set_voice_state(listening: bool) -> None:
        nonlocal voice_state_timer, voice_state_generation
        with voice_state_lock:
            voice_state_generation += 1
            generation = voice_state_generation
            if voice_state_timer is not None:
                voice_state_timer.cancel()
                voice_state_timer = None
            stream.emit(
                "listening" if listening else "idle",
                detail={"label": "push-to-talk"} if listening else {"label": "ready"},
                active={"microphone": listening},
            )
            if listening:
                def expire_listening() -> None:
                    nonlocal voice_state_timer
                    with voice_state_lock:
                        if generation != voice_state_generation:
                            return
                        voice_state_timer = None
                        stream.emit(
                            "idle", detail={"label": "ready"}, active={"microphone": False}
                        )

                # If a tab disappears without its stop request, do not leave a
                # stale microphone indicator active indefinitely.
                voice_state_timer = threading.Timer(20.0, expire_listening)
                voice_state_timer.daemon = True
                voice_state_timer.start()

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
            elif path == "/today":
                self._today()
            elif path == "/healthz":
                self._send(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok\n")
            elif path == "/control-state" and control_path is not None:
                payload = json.dumps(RuntimeControl(control_path).read(), sort_keys=True).encode()
                self._send(HTTPStatus.OK, "application/json", payload)
            elif path == "/improvements" and control_path is not None:
                ledger = ImprovementLedger(control_path.parent)
                payload = json.dumps(ledger.list(), sort_keys=True).encode()
                self._send(HTTPStatus.OK, "application/json", payload)
            elif path.startswith("/api/"):
                if not self._same_origin_api():
                    self._send(HTTPStatus.FORBIDDEN, "text/plain; charset=utf-8", b"same-origin access required\n")
                elif database_path is None:
                    self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain; charset=utf-8", b"local database unavailable\n")
                else:
                    self._api_get(path, urlparse(self.path).query)
            else:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")

        def do_POST(self) -> None:  # noqa: N802
            request_path = urlparse(self.path).path
            allowed_posts = {
                "/control", "/improvements", "/api/directives",
                "/api/preferences", "/api/conversations", "/api/voice-preview", "/api/chat",
                "/api/voice-model", "/api/transcribe", "/api/voice-state",
            }
            if request_path not in allowed_posts or control_path is None:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")
                return
            host = self.headers.get("Host", "")
            expected_origin = f"http://{host}"
            if (
                not _is_loopback_host_header(host)
                or self.headers.get("Origin") != expected_origin
                or self.headers.get("Sec-Fetch-Site", "same-origin") not in {"same-origin", "none"}
                or self.headers.get_content_type() != "application/json"
            ):
                self._send(HTTPStatus.FORBIDDEN, "text/plain; charset=utf-8", b"same-origin JSON required\n")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body_limit = 15 * 1024 * 1024 if request_path == "/api/transcribe" else (
                    65536 if request_path.startswith("/api/") else 1024
                )
                if not 1 <= length <= body_limit:
                    raise ValueError("invalid body size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                if request_path == "/control":
                    result = RuntimeControl(control_path).update(body)
                elif request_path == "/improvements":
                    action = body.get("action")
                    candidate_id = body.get("id")
                    if not isinstance(candidate_id, str) or action not in {"approve", "reject", "promote"}:
                        raise ValueError("provide a candidate id and approve, reject, or promote action")
                    ledger = ImprovementLedger(control_path.parent)
                    if action == "promote":
                        from claireai.config import Settings
                        ledger.promote(candidate_id, active_checkpoint=Settings(data_dir=control_path.parent).resolved_model_checkpoint)
                    else:
                        ledger.decide(candidate_id, "approved" if action == "approve" else "rejected")
                    result = ledger.list()
                elif database_path is not None:
                    result = self._api_post(request_path, body)
                else:
                    raise ValueError("local database unavailable")
            except (ValueError, OSError, sqlite3.Error, ImprovementError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, "text/plain; charset=utf-8", str(exc).encode()[:512])
                return
            except BrainStopped as exc:
                self._send(HTTPStatus.CONFLICT, "text/plain; charset=utf-8", str(exc).encode()[:512])
                return
            except BrainError as exc:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain; charset=utf-8", str(exc).encode()[:512])
                return
            except SpeechRecognitionUnavailable as exc:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain; charset=utf-8", str(exc).encode()[:512])
                return
            self._send(HTTPStatus.OK, "application/json", json.dumps(result, sort_keys=True).encode())

        def _same_origin_api(self) -> bool:
            host = self.headers.get("Host", "")
            if not _is_loopback_host_header(host):
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{host}":
                return False
            return self.headers.get("Sec-Fetch-Site", "same-origin") in {"same-origin", "none"}

        def _api_get(self, path: str, query: str) -> None:
            assert database_path is not None
            if path == "/api/voice-status":
                from claireai.voice_input import transcriber_for
                data_dir = control_path.parent if control_path is not None else Path(database_path).parent.parent
                transcriber = transcriber_for(str(data_dir / "models" / "voice" / "asr"))
                payload = {
                    "runtime_available": transcriber.runtime_available,
                    "model_installed": transcriber.installed,
                    "model": "Whisper tiny (multilingual, local CPU)",
                }
                self._send(HTTPStatus.OK, "application/json", json.dumps(payload).encode())
                return
            conn = open_read_only(database_path)
            if conn is None:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain; charset=utf-8", b"local database unavailable\n")
                return
            try:
                if path == "/api/conversations":
                    params = parse_qs(query)
                    conversation_id = params.get("id", [None])[0]
                    repository = ConversationRepository(conn)
                    payload = (
                        [message.__dict__ for message in repository.messages(conversation_id)]
                        if conversation_id
                        else repository.recent()
                    )
                elif path == "/api/directives":
                    payload = DirectiveRepository(conn).list()
                elif path == "/api/preferences":
                    data_dir = control_path.parent if control_path is not None else Path(database_path).parent.parent
                    payload = PreferenceRepository(conn).effective(Settings(data_dir=data_dir))
                else:
                    self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")
                    return
                self._send(HTTPStatus.OK, "application/json", json.dumps(payload, sort_keys=True).encode())
            except ValueError as exc:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", str(exc).encode()[:512])
            finally:
                conn.close()

        def _api_post(self, path: str, body: dict[str, Any]) -> Any:
            assert database_path is not None
            database = Database(database_path)
            conn = database.connect()
            try:
                if path == "/api/voice-state":
                    listening = body.get("listening")
                    if type(listening) is not bool:
                        raise ValueError("listening must be true or false")
                    if listening and RuntimeControl(control_path).read()["shutdown"]:
                        raise BrainStopped("ClaireAI is stopped in the local control panel.")
                    set_voice_state(listening)
                    return {"listening": listening}
                if path in {"/api/voice-model", "/api/transcribe"}:
                    from claireai.voice_input import (
                        MAX_AUDIO_BYTES,
                        SUPPORTED_AUDIO_TYPES,
                        SpeechRecognitionUnavailable,
                        transcriber_for,
                    )
                    if RuntimeControl(control_path).read()["shutdown"]:
                        raise BrainStopped("ClaireAI is stopped in the local control panel.")
                    data_dir = Path(control_path).parent
                    transcriber = transcriber_for(str(data_dir / "models" / "voice" / "asr"))
                    if path == "/api/voice-model":
                        transcriber.install()
                        return {
                            "installed": transcriber.installed,
                            "model": "Whisper tiny (multilingual, local CPU)",
                        }
                    if not transcriber.installed:
                        raise SpeechRecognitionUnavailable(
                            "install the local speech model from the voice panel first"
                        )
                    mime_type = body.get("content_type")
                    encoded = body.get("audio_base64")
                    if not isinstance(mime_type, str) or mime_type.split(";", 1)[0].lower() not in SUPPORTED_AUDIO_TYPES:
                        raise ValueError("unsupported browser audio format")
                    if not isinstance(encoded, str) or len(encoded) > ((MAX_AUDIO_BYTES + 2) // 3) * 4 + 8:
                        raise ValueError("audio payload is invalid or too large")
                    try:
                        audio = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ValueError("audio payload is not valid base64") from exc
                    if len(audio) > MAX_AUDIO_BYTES:
                        raise ValueError("audio payload is too large")
                    text = transcriber.transcribe(
                        audio,
                        content_type=mime_type.split(";", 1)[0].lower(),
                    )
                    if RuntimeControl(control_path).read()["shutdown"]:
                        raise BrainStopped("ClaireAI was stopped during speech transcription.")
                    return {"text": text}
                if path == "/api/chat":
                    message = body.get("message")
                    conversation_id = body.get("conversation_id")
                    if not isinstance(message, str) or not message.strip() or len(message) > 6000:
                        raise ValueError("message must contain 1–6000 characters")
                    if conversation_id is not None and not isinstance(conversation_id, str):
                        raise ValueError("conversation id must be a string")
                    data_dir = Path(control_path).parent if control_path is not None else Path(database_path).parent.parent
                    if RuntimeControl(data_dir / "runtime-control.json").read()["shutdown"]:
                        raise BrainStopped("ClaireAI is stopped in the local control panel.")
                    settings = PreferenceRepository(conn).effective(Settings(data_dir=data_dir))
                    settings = Settings(_env_file=None, data_dir=data_dir, **settings)
                    brain = ScratchBrain(
                        checkpoint=settings.resolved_model_checkpoint,
                        max_output_tokens=settings.max_output_tokens,
                        control_path=settings.control_state_path,
                    )

                    async def run_chat():
                        health = await brain.health()
                        if not health.available or not health.model_present:
                            raise BrainError(health.detail or "local scratch model is unavailable")
                        try:
                            from claireai.cli import ask, _speak_answer
                            turn = await ask(
                                settings=settings,
                                conn=conn,
                                brain=brain,
                                question=message.strip(),
                                device="local-ui",
                                conversation_id=conversation_id,
                            )
                            if settings.voice_enabled:
                                from argparse import Namespace
                                _speak_answer(
                                    args=Namespace(device="local-ui", voice=None, voice_pitch=None, voice_rate=None),
                                    settings=settings,
                                    conn=conn,
                                    task_id=turn.task_id,
                                    text=turn.content,
                                )
                            return {
                                "conversation_id": turn.conversation_id,
                                "task_id": turn.task_id,
                                "model": turn.model,
                                "content": turn.content,
                            }
                        finally:
                            await brain.aclose()

                    import asyncio
                    return asyncio.run(run_chat())
                if path == "/api/preferences":
                    return PreferenceRepository(conn).update(body)
                if path == "/api/voice-preview":
                    preferences = PreferenceRepository(conn).effective(
                        Settings(data_dir=Path(database_path).parent.parent)
                    )
                    from claireai.voice import VoiceUnavailable, speak_text
                    try:
                        engine = speak_text(
                            "Hello Ange. This is a local ClaireAI voice preview.",
                            pitch=preferences["voice_pitch"],
                            rate=preferences["voice_rate"],
                        )
                    except VoiceUnavailable as exc:
                        raise ValueError(str(exc)) from exc
                    return {"engine": engine, "previewed": True}
                if path == "/api/directives":
                    repository = DirectiveRepository(conn)
                    action = body.get("action", "save")
                    if action == "delete":
                        directive_id = body.get("id")
                        if not isinstance(directive_id, str) or not repository.delete(directive_id):
                            raise ValueError("directive does not exist")
                    elif action == "save":
                        title, content = body.get("title"), body.get("content")
                        if not isinstance(title, str) or not isinstance(content, str):
                            raise ValueError("directive title and content are required")
                        directive_id = body.get("id")
                        if directive_id is not None and not isinstance(directive_id, str):
                            raise ValueError("directive id must be a string")
                        enabled = body.get("enabled", True)
                        repository.save(
                            directive_id=directive_id,
                            title=title,
                            content=content,
                            enabled=enabled,
                        )
                    else:
                        raise ValueError("directive action must be save or delete")
                    return repository.list()
                if path == "/api/conversations":
                    conversation_id = body.get("id")
                    if not isinstance(conversation_id, str):
                        raise ValueError("conversation id is required")
                    if not ConversationRepository(conn).delete(conversation_id):
                        raise ValueError("conversation does not exist")
                    return ConversationRepository(conn).recent()
                raise ValueError("unsupported local API action")
            finally:
                conn.close()

        def _send(
            self,
            status: HTTPStatus,
            content_type: str,
            body: bytes,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _today(self) -> None:
            if database_path is None:
                page = render_unavailable(
                    "No task log is configured. Start the preview with --db pointing at "
                    "the ClaireAI database."
                )
            else:
                conn = open_read_only(database_path)
                if conn is None:
                    page = render_unavailable(f"No database at {database_path} yet.")
                else:
                    try:
                        page = render_today(conn)
                    except sqlite3.Error as exc:
                        page = render_unavailable(f"The database could not be read: {exc}")
                    finally:
                        conn.close()
            self._send(
                HTTPStatus.OK,
                "text/html; charset=utf-8",
                page.encode("utf-8"),
                {"Content-Security-Policy": TODAY_CSP},
            )

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
    database_path: Path | None = None,
    control_path: Path | None = None,
) -> ThreadingHTTPServer:
    """Create (but do not start) the preview server.

    ``database_path`` enables the read-only ``/today`` page; without it that path
    explains that no log is configured instead of guessing at one.
    """
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"the presence preview binds to loopback only, refusing host {host!r}; "
            "a LAN-exposed renderer needs the paired API and a threat review"
        )
    html = html_path.read_bytes()
    handler = _handler_factory(
        stream,
        html,
        html_path.name,
        Path(database_path) if database_path else None,
        Path(control_path) if control_path else None,
    )
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
    parser.add_argument("--data-dir", help="ClaireAI data directory shared with the CLI runtime")
    parser.add_argument("--demo", action="store_true", help="cycle fake states for a visual check")
    parser.add_argument(
        "--html",
        default=str(Path(__file__).resolve().parents[3] / "desktop" / "presence" / "index.html"),
    )
    parser.add_argument(
        "--db",
        help="database for the read-only /today page (default: the configured one if it exists)",
    )
    args = parser.parse_args(argv)

    from claireai.config import Settings

    settings = Settings(data_dir=Path(args.data_dir).expanduser() if args.data_dir else None)
    settings.ensure_directories()
    database_path = Path(args.db).expanduser() if args.db else settings.database_path
    Database(database_path).migrate()
    control_path = settings.control_state_path

    stream = PresenceStream()
    stop = threading.Event()
    if args.demo:
        threading.Thread(target=demo_loop, args=(stream,), kwargs={"stop": stop}, daemon=True).start()

    server = build_server(
        stream,
        html_path=Path(args.html),
        host=args.host,
        port=args.port,
        database_path=database_path,
        control_path=control_path,
    )
    host, port = server.server_address[:2]
    print(f"presence preview: http://{host}:{port}/  ({len(STATES)} states)")
    if database_path:
        print(f"today (read-only): http://{host}:{port}/today  ← {database_path}")
    else:
        print("today page disabled: no database found; pass --db PATH to enable it")
    if args.demo:
        print("demo mode: fake state changes every 1.5 s")
    print(f"local controls: {control_path}")
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
