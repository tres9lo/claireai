"""The corpus harvester: what it keeps, and what it refuses."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "brain-lab" / "harvest_corpus.py"


def _module():
    spec = importlib.util.spec_from_file_location("harvest_corpus", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


harvest = _module()


def test_html_is_reduced_to_readable_paragraphs() -> None:
    page = (
        "<html><head><style>body{color:red}</style><script>alert(1)</script></head>"
        "<body><nav>menu</nav>"
        "<p>This paragraph is long enough to be kept, because it runs well past "
        "the minimum length a training sample needs.</p>"
        "<p>short</p>"
        "<footer>footer text</footer></body></html>"
    )
    text = harvest.html_to_text(page)

    assert "long enough to be kept" in text, repr(text)
    assert "short" not in text
    assert "footer" not in text
    assert "color:red" not in text
    assert "alert" not in text
    assert "menu" not in text


def test_non_http_urls_are_refused() -> None:
    with pytest.raises(harvest.HarvestError):
        harvest.robots_allows("file:///etc/passwd")
    with pytest.raises(harvest.HarvestError):
        harvest.robots_allows("ftp://example.com/x.txt")


def _serve(routes: dict[str, tuple[int, str]]) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN002, ANN003
            return

        def do_GET(self):  # noqa: N802
            status, body = routes.get(self.path, (404, "missing"))
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def test_a_disallowed_path_is_not_fetched() -> None:
    server, base = _serve(
        {
            "/robots.txt": (200, "User-agent: *\nDisallow: /private\n"),
            "/private/x.txt": (200, "secret " * 40),
        }
    )
    try:
        assert harvest.robots_allows(f"{base}/public/x.txt") is True
        assert harvest.robots_allows(f"{base}/private/x.txt") is False
    finally:
        server.shutdown()


def test_an_unreachable_robots_file_is_not_permission() -> None:
    server, base = _serve({})
    try:
        assert harvest.robots_allows(f"{base}/anything.txt") is False
    finally:
        server.shutdown()


def test_a_corpus_is_written_with_a_manifest(tmp_path) -> None:
    body = "\n\n".join(
        f"Paragraph {index} explains a small, verifiable fact about local software." * 2
        for index in range(20)
    )
    manifest = harvest.write_corpus("demo", [("https://example.com/a", body)], tmp_path)

    target = tmp_path / "demo"
    assert (target / "text" / "train" / "train.txt").is_file()
    assert (target / "text" / "validation" / "validation.txt").is_file()
    assert (target / "manifest.json").is_file()

    stored = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert stored["documents"][0]["sha256"] == manifest["documents"][0]["sha256"]
    assert stored["documents"][0]["license"].startswith("unverified")


def test_too_little_text_is_refused(tmp_path) -> None:
    with pytest.raises(harvest.HarvestError):
        harvest.write_corpus("empty", [("https://example.com", "tiny")], tmp_path)


def test_the_download_is_bounded() -> None:
    body = "x" * 5000
    server, base = _serve({"/big.txt": (200, body)})
    try:
        with pytest.raises(harvest.HarvestError):
            harvest.fetch(f"{base}/big.txt", max_bytes=1000)
        assert harvest.fetch(f"{base}/big.txt", max_bytes=10000) == body
    finally:
        server.shutdown()
