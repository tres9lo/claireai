"""The browser tools: bounded fetches, sandboxed downloads, no surprises.

Everything here runs against httpx.MockTransport — no real network, no browser.
That is honest about what these tests prove: our request handling, parsing,
limits and refusals, not that a live site renders.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

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


def _session(*, body: str = PAGE, status: int = 200, content_type: str = "text/html") -> BrowserSession:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            text=body,
            headers={"content-type": content_type},
        )

    return BrowserSession(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _context(root: Path, session: BrowserSession | None) -> ToolContext:
    return ToolContext(
        sandbox=Sandbox(roots=(root.resolve(),)),
        browser=session if session is not None else BrowserSession(client=None),
    )


# --- parsing ---------------------------------------------------------------


def test_readable_text_keeps_prose_and_drops_tags_and_scripts() -> None:
    title, text = readable_text(PAGE)
    assert title == "Invoices"
    assert "March invoices" in text
    assert "Total due: 41.90 EUR" in text
    assert "secret" not in text  # script contents are dropped
    assert "<p>" not in text


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
