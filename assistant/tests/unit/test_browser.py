"""The browser tools: bounded fetches, sandboxed downloads, no surprises.

Everything here runs against httpx.MockTransport — no real network, no browser.
That is honest about what these tests prove: our request handling, parsing,
limits and refusals, not that a live site renders.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

import claireai.tools.browser as browser_module
from claireai.permissions import PermissionDenied, Sandbox
from claireai.routing import route
from claireai.tools import default_registry
from claireai.tools.base import ToolContext, ToolError
from claireai.tools.browser import (
    BrowserSession,
    DownloadFileArgs,
    OpenUrlArgs,
    PageTextArgs,
    SearchWebArgs,
    download_file,
    get_page_text,
    open_url,
    readable_text,
    result_url,
    search_web,
)

PAGE = """<html><head><title>Invoices</title>
<script>var secret = "ignore me";</script></head>
<body><h1>March invoices</h1><p>Total due: 41.90 EUR</p></body></html>"""

SEARCH = """<div class="result">
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">First result</a>
</div><div class="result">
<a rel="nofollow" class="result__a" href="https://example.com/b">Second result</a>
</div>"""


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    return folder


PUBLIC = ipaddress.ip_address("93.184.216.34")


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the egress check off the network for this file only.

    Literal addresses resolve to themselves, so the blocked-address tests still
    exercise the real check; local names keep resolving to loopback; every other
    hostname becomes one fixed public address.
    """

    def fake(host: str, port: int) -> tuple[ipaddress._BaseAddress, ...]:
        del port
        try:
            return (ipaddress.ip_address(host),)
        except ValueError:
            if host.lower().rstrip(".") in {"localhost", "localhost.localdomain", "ip6-localhost"}:
                return (ipaddress.ip_address("127.0.0.1"),)
            return (PUBLIC,)

    monkeypatch.setattr(browser_module, "_resolved_addresses", fake)


_CREATED: list[BrowserSession] = []


@pytest.fixture(autouse=True)
def close_sessions():
    """Close every client this test opened, so nothing leaks into the next one."""
    yield
    while _CREATED:
        _CREATED.pop().close()


def _session(*, body: str = PAGE, status: int = 200, content_type: str = "text/html") -> BrowserSession:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            text=body,
            headers={"content-type": content_type},
        )

    session = BrowserSession(transport=httpx.MockTransport(handler))
    _CREATED.append(session)
    return session


def _context(root: Path, session: BrowserSession | None) -> ToolContext:
    return ToolContext(
        sandbox=Sandbox(roots=(root.resolve(),)),
        browser=session if session is not None else BrowserSession(),
    )


# --- parsing ---------------------------------------------------------------


def test_readable_text_keeps_prose_and_drops_tags_and_scripts() -> None:
    title, text = readable_text(PAGE)
    assert title == "Invoices"
    assert "March invoices" in text
    assert "Total due: 41.90 EUR" in text
    assert "secret" not in text  # script contents are dropped
    assert "<p>" not in text


def test_readable_text_drops_navigation_and_footer_furniture() -> None:
    _, text = readable_text(
        "<nav>Main page Contents Random article</nav>"
        "<main><p>The capital is Kigali.</p></main>"
        "<footer>Privacy policy</footer>"
    )
    assert "Kigali" in text
    assert "Random article" not in text
    assert "Privacy policy" not in text


def test_readable_text_is_clipped() -> None:
    _, text = readable_text("<p>" + ("word " * 500) + "</p>", limit=50)
    assert len(text) <= 50


# --- open_url --------------------------------------------------------------


def test_open_url_returns_text_and_remembers_the_page(root: Path) -> None:
    session = _session()
    result = open_url(_context(root, session), OpenUrlArgs(url="https://example.com/inv"))

    assert "March invoices" in result.text
    assert result.detail["title"] == "Invoices"
    assert session.url.endswith("/inv")
    assert "Total due" in session.text


def test_open_url_only_accepts_http_and_https(root: Path) -> None:
    context = _context(root, _session())
    for bad in ("file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", ""):
        with pytest.raises(ToolError, match="url|http"):
            open_url(context, OpenUrlArgs(url=bad))


def test_open_url_reports_an_http_error(root: Path) -> None:
    with pytest.raises(ToolError, match="HTTP 404"):
        open_url(_context(root, _session(status=404)), OpenUrlArgs(url="https://example.com/none"))


def test_open_url_refuses_an_oversized_page(root: Path, monkeypatch) -> None:
    from claireai.tools import browser as browser_module

    monkeypatch.setattr(browser_module, "MAX_PAGE_BYTES", 10)
    with pytest.raises(ToolError, match="limit"):
        open_url(_context(root, _session()), OpenUrlArgs(url="https://example.com/big"))


def test_open_url_is_a_dry_run_without_fetching_side_effects(root: Path) -> None:
    context = ToolContext(sandbox=Sandbox(roots=(root.resolve(),)), browser=_session(), dry_run=True)
    result = open_url(context, OpenUrlArgs(url="https://example.com/x"))
    assert result.text.startswith("[dry run] would open")
    assert result.detail["dry_run"] is True


# --- get_page_text ---------------------------------------------------------


def test_get_page_text_needs_an_open_page(root: Path) -> None:
    with pytest.raises(ToolError, match="no page is open"):
        get_page_text(_context(root, _session()), PageTextArgs())


def test_get_page_text_reads_what_was_opened(root: Path) -> None:
    session = _session()
    context = _context(root, session)
    open_url(context, OpenUrlArgs(url="https://example.com/inv"))

    result = get_page_text(context, PageTextArgs(max_chars=200))
    assert "March invoices" in result.text
    assert result.detail["url"].endswith("/inv")


def test_browsing_is_unavailable_without_a_session(root: Path) -> None:
    class NoBrowser(ToolContext):
        pass

    context = ToolContext(sandbox=Sandbox(roots=(root.resolve(),)))
    with pytest.raises(ToolError, match="not available"):
        open_url(context, OpenUrlArgs(url="https://example.com/x"))


# --- search_web ------------------------------------------------------------


def test_search_web_parses_titles_and_links(root: Path) -> None:
    result = search_web(_context(root, _session(body=SEARCH)), SearchWebArgs(query="invoices march"))

    assert "https://example.com/a" in result.text
    assert "First result" in result.text
    assert result.detail["shown"] == 2


def test_search_web_unwraps_the_redirect_to_the_real_url() -> None:
    # The endpoint hands back its own redirect; the model needs the target.
    assert (
        result_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1")
        == "https://example.com/a"
    )
    assert result_url("https://example.com/b") == "https://example.com/b"


def test_search_web_says_so_when_nothing_parses(root: Path) -> None:
    with pytest.raises(ToolError, match="no parseable results"):
        search_web(_context(root, _session(body="<html>nothing here</html>")), SearchWebArgs(query="x"))


def test_search_web_respects_max_results(root: Path) -> None:
    result = search_web(_context(root, _session(body=SEARCH)), SearchWebArgs(query="q", max_results=1))
    assert result.detail["shown"] == 1
    assert "https://example.com/b" not in result.text


# --- download_file ---------------------------------------------------------


def test_download_file_writes_inside_the_allow_list(root: Path) -> None:
    target = root / "invoice.bin"
    result = download_file(
        _context(root, _session(body="payload")),
        DownloadFileArgs(url="https://example.com/f", path=str(target)),
    )

    assert target.read_bytes() == b"payload"
    assert result.detail["bytes"] == len("payload")


def test_download_file_refuses_to_overwrite(root: Path) -> None:
    target = root / "keep.txt"
    target.write_text("mine", encoding="utf-8")

    with pytest.raises(ToolError, match="already exists"):
        download_file(
            _context(root, _session()),
            DownloadFileArgs(url="https://example.com/f", path=str(target)),
        )
    assert target.read_text(encoding="utf-8") == "mine"


def test_download_file_cannot_reach_outside_the_allow_list(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    with pytest.raises(PermissionDenied):
        download_file(
            _context(root, _session()),
            DownloadFileArgs(url="https://example.com/f", path=str(outside)),
        )
    assert not outside.exists()


def test_download_file_is_a_dry_run_without_writing(root: Path) -> None:
    target = root / "dry.bin"
    context = ToolContext(sandbox=Sandbox(roots=(root.resolve(),)), browser=_session(), dry_run=True)

    result = download_file(context, DownloadFileArgs(url="https://example.com/f", path=str(target)))
    assert result.text.startswith("[dry run] would download")
    assert not target.exists()


def test_download_file_refuses_a_missing_parent(root: Path) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        download_file(
            _context(root, _session()),
            DownloadFileArgs(url="https://example.com/f", path=str(root / "nope" / "x.bin")),
        )


# --- egress policy ----------------------------------------------------------

NON_PUBLIC = (
    "127.0.0.1",
    "127.0.0.53",
    "0.0.0.0",
    "::1",
    "10.0.0.1",
    "172.20.0.5",
    "192.168.1.10",
    "169.254.169.254",
    "fd12:3456::1",
    "fe80::1",
    "[::ffff:127.0.0.1]",
    "100.64.0.7",
    "64:ff9b::7f00:1",
    "64:ff9b:1::1",
    "2002:7f00:1::",
)


def _recording_session(requests: list[str]) -> BrowserSession:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})

    session = BrowserSession(transport=httpx.MockTransport(handler))
    _CREATED.append(session)
    return session


@pytest.mark.parametrize("host", NON_PUBLIC)
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_the_browser_refuses_this_machine_and_the_local_network(root: Path, host: str, scheme: str) -> None:
    requests: list[str] = []
    context = _context(root, _recording_session(requests))
    url = f"{scheme}://{host}/secret"
    with pytest.raises(ToolError, match="not a public address"):
        open_url(context, OpenUrlArgs(url=url))
    assert requests == [], "the request must be refused before it is sent"


def test_download_file_refuses_a_private_address(root: Path) -> None:
    requests: list[str] = []
    context = _context(root, _recording_session(requests))
    with pytest.raises(ToolError, match="not a public address"):
        download_file(
            context, DownloadFileArgs(url="http://192.168.0.10/file.bin", path=str(root / "f.bin"))
        )
    assert requests == []
    assert not (root / "f.bin").exists()


def test_a_public_page_may_redirect_to_another_public_page(root: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "https://example.com/new"})
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})

    session = BrowserSession(transport=httpx.MockTransport(handler))
    _CREATED.append(session)
    result = open_url(_context(root, session), OpenUrlArgs(url="https://example.com/old"))
    assert "March invoices" in result.text
    assert session.url == "https://example.com/new"


def test_a_redirect_to_a_private_address_is_refused(root: Path) -> None:
    """Checking only the first URL would let a public page bounce to loopback."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1:8765/api/state"})

    session = BrowserSession(transport=httpx.MockTransport(handler))
    _CREATED.append(session)
    with pytest.raises(ToolError, match="not a public address"):
        open_url(_context(root, session), OpenUrlArgs(url="https://example.com/goto"))
    # The public hop happened; loopback never did.
    assert seen == ["https://example.com/goto"]


def test_a_redirect_chain_that_never_settles_is_refused(root: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        target = "https://example.com/b" if request.url.path == "/a" else "https://example.com/a"
        return httpx.Response(302, headers={"location": target})

    session = BrowserSession(transport=httpx.MockTransport(handler))
    _CREATED.append(session)
    with pytest.raises(ToolError, match="too many redirects"):
        open_url(_context(root, session), OpenUrlArgs(url="https://example.com/a"))


def test_a_hostname_that_resolves_to_loopback_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        browser_module, "_resolved_addresses", lambda host, port: (ipaddress.ip_address("127.0.0.1"),)
    )
    requests: list[str] = []
    with pytest.raises(ToolError, match="not a public address"):
        open_url(
            _context(root, _recording_session(requests)),
            OpenUrlArgs(url="http://internal.example.test/"),
        )
    assert requests == []


def test_dns_rebinding_is_pinned_to_the_first_public_answer(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second DNS answer must not turn a checked request into a private one."""
    answers = [
        (PUBLIC,),
        (ipaddress.ip_address("10.0.0.7"),),
    ]

    def rebinding(host: str, port: int) -> tuple[ipaddress._BaseAddress, ...]:
        del host, port
        return answers.pop(0)

    monkeypatch.setattr(browser_module, "_resolved_addresses", rebinding)
    requests: list[str] = []
    session = _recording_session(requests)

    open_url(_context(root, session), OpenUrlArgs(url="https://rebind.example/page"))

    # The request may only be sent to the address that passed the check. There
    # must be no second lookup that could return the private address.
    assert requests == ["https://rebind.example/page"]
    assert session.pins[("rebind.example", 443)] == (str(PUBLIC),)
    assert answers == [(ipaddress.ip_address("10.0.0.7"),)]


def test_every_public_answer_is_pinned_for_failover(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second = ipaddress.ip_address("93.184.216.35")
    monkeypatch.setattr(
        browser_module,
        "_resolved_addresses",
        lambda host, port: (PUBLIC, second),
    )
    session = _recording_session([])

    open_url(_context(root, session), OpenUrlArgs(url="https://example.com/"))

    assert session.pins[("example.com", 443)] == (str(PUBLIC), str(second))


def test_the_pinned_backend_connects_to_the_checked_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected: list[tuple[str, int]] = []

    class FakeSocket:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def settimeout(self, timeout) -> None:  # noqa: ANN001
            pass

        def connect(self, address) -> None:  # noqa: ANN001
            connected.append(address)

        def setsockopt(self, *args) -> None:  # noqa: ANN002
            pass

    monkeypatch.setattr(browser_module.socket, "socket", FakeSocket)
    backend = browser_module._PinnedBackend({("rebind.example", 80): (str(PUBLIC),)})
    backend.connect_tcp("rebind.example", 80)

    assert connected == [(str(PUBLIC), 80)]


def test_the_pinned_backend_fails_over_to_the_next_checked_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = str(PUBLIC)
    second = "93.184.216.35"
    attempts: list[tuple[str, int]] = []

    class FakeSocket:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.rejected = False

        def settimeout(self, timeout) -> None:  # noqa: ANN001
            pass

        def connect(self, address) -> None:  # noqa: ANN001
            attempts.append(address)
            if address[0] == public:
                self.rejected = True
                raise OSError("first address is down")

        def setsockopt(self, *args) -> None:  # noqa: ANN002
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(browser_module.socket, "socket", FakeSocket)
    backend = browser_module._PinnedBackend({("rebind.example", 80): (public, second)})
    backend.connect_tcp("rebind.example", 80)

    assert attempts == [(public, 80), (second, 80)]


def test_the_pinned_backend_skips_an_unavailable_address_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[str, int]] = []

    class FakeSocket:
        def __init__(self, family, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
            if family == socket.AF_INET6:
                raise OSError("IPv6 is unavailable")

        def settimeout(self, timeout) -> None:  # noqa: ANN001
            pass

        def connect(self, address) -> None:  # noqa: ANN001
            attempts.append(address)

        def setsockopt(self, *args) -> None:  # noqa: ANN002
            pass

    monkeypatch.setattr(browser_module.socket, "socket", FakeSocket)
    backend = browser_module._PinnedBackend(
        {
            ("rebind.example", 80): (
                "2606:4700:4700::1111",
                str(PUBLIC),
            )
        }
    )
    backend.connect_tcp("rebind.example", 80)

    assert attempts == [(str(PUBLIC), 80)]


def test_the_pinned_transport_uses_the_checked_address_over_a_real_socket() -> None:
    """Exercise the real httpcore socket path, not a mocked transport."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b"pinned"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: ANN001, ANN002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    client = httpx.Client(
        transport=browser_module._PinnedTransport(
            {("pinned.example", port): ("127.0.0.1",)}
        ),
        trust_env=False,
    )
    try:
        response = client.get(f"http://pinned.example:{port}/")
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert response.status_code == 200
    assert response.text == "pinned"


def test_the_default_session_always_builds_the_pinned_transport() -> None:
    session = BrowserSession()
    _CREATED.append(session)

    client = session.http()

    assert isinstance(client._transport, browser_module._PinnedTransport)
    assert client.follow_redirects is False
    assert client.trust_env is False


def test_an_unresolvable_host_is_refused_without_fetching(root: Path) -> None:
    def boom(host: str, port: int) -> tuple[ipaddress._BaseAddress, ...]:
        raise ToolError(f"could not resolve {host!r}")

    import claireai.tools.browser as module

    original = module._resolved_addresses
    module._resolved_addresses = boom
    try:
        with pytest.raises(ToolError, match="could not resolve"):
            open_url(_context(root, _session()), OpenUrlArgs(url="https://nope.invalid/"))
    finally:
        module._resolved_addresses = original


# --- wiring ----------------------------------------------------------------


def test_the_registry_marks_web_tools_as_needing_approval() -> None:
    registry = default_registry(browser_enabled=True)
    for name in ("open_url", "search_web", "download_file"):
        assert registry.get(name).risk == "needs_approval"
    # Reading a page already fetched in this task touches nothing.
    assert registry.get("get_page_text").risk == "safe"
    # Only the download mutates the filesystem.
    assert registry.get("download_file").mutates is True
    assert registry.get("open_url").mutates is False


def test_web_requests_route_to_the_browser_tools() -> None:
    registry = default_registry(browser_enabled=True)
    for text in (
        "open https://example.com and tell me what it says",
        "search the web for python 3.13 release notes",
    ):
        result = route(text, registry)
        assert result.intent == "browse", text
        assert "open_url" in result.tools


def test_opening_a_local_file_still_routes_to_the_file_tools() -> None:
    # "open" is shared with reading a file; the object has to decide.
    result = route("open the file notes.txt", default_registry(browser_enabled=True))
    assert result.intent == "read"
    assert "read_text_file" in result.tools
