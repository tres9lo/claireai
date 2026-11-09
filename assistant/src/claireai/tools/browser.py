"""Browser tools (Month 1, Week 3).

The roadmap names Playwright here. This implements the same four tools —
``open_url``, ``get_page_text``, ``search_web``, ``download_file`` — over plain
HTTP, because Playwright means downloading a Chromium build (and its system
dependencies) onto a machine that has neither the RAM nor the package access
right now. What that costs, stated plainly: **no JavaScript runs**, so a page
that renders its content client-side comes back empty. What it buys: no new
dependency, no browser profile to leak, and a tool surface that is already
approval-gated, size-capped and testable without a real network.

When Playwright is installed, only this module changes: the tool names, their
arguments and their risk levels are what the model and the approvals see.

Every fetch is bounded (bytes and characters), only http(s) is accepted, and
nothing here writes to disk except ``download_file``, which goes through the
same sandbox as every other write.

**Egress policy.** These tools run on the same machine as the presence server
and the owner's data, and the page they read is written by someone else.  A URL
that points at this machine, the local network or a link-local metadata endpoint
is therefore refused before the request is made, and the check is repeated for
every redirect hop — checking only the first URL would let a public page bounce
the request to ``127.0.0.1``.  Resolution happens here rather than being left to
httpx so that a hostname cannot resolve to a private address behind the check.
"""

from __future__ import annotations

import html as html_module
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx
from pydantic import Field

from claireai.tools.base import ToolContext, ToolError, ToolResult
from claireai.tools.files import _Args

MAX_PAGE_BYTES = 2_000_000
MAX_TEXT_CHARS = 4000
MAX_DOWNLOAD_BYTES = 20_000_000
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = frozenset({"http", "https"})
USER_AGENT = "ClaireAI/0.1 (local assistant; +https://github.com/tres9lo/claireai)"


class _ReadableText(HTMLParser):
    """Collect readable text: no tags, no scripts, no styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skipping = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style", "noscript"}:
            self._skipping += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "br", "li", "div", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skipping:
            self._skipping -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif not self._skipping:
            self.parts.append(text)


def readable_text(html: str, *, limit: int = MAX_TEXT_CHARS) -> tuple[str, str]:
    """Return ``(title, text)`` from HTML, clipped to ``limit`` characters."""
    parser = _ReadableText()
    parser.feed(html)
    body = re.sub(r"[ \t]+", " ", "".join(
        part if part == "\n" else f"{part} " for part in parser.parts
    ))
    body = re.sub(r"\n\s*\n+", "\n", body).strip()
    return parser.title, body[:limit]


@dataclass
class BrowserSession:
    """One task's browsing state: the page last opened, and the client.

    A session per task, not per call: ``get_page_text`` reads what
    ``open_url`` fetched, and nothing leaks between tasks.
    """

    timeout: float = 15.0
    client: httpx.Client | None = None
    url: str = ""
    title: str = ""
    text: str = ""
    fetched: int = 0

    def http(self) -> httpx.Client:
        if self.client is None:
            # Redirects are followed by _get, one hop at a time, so each new
            # destination passes the egress check.  Letting httpx follow them
            # here would fetch the first hidden-until-followed target.
            self.client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,
                headers={"User-Agent": USER_AGENT},
            )
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def _session(context: ToolContext) -> BrowserSession:
    session = getattr(context, "browser", None)
    if not isinstance(session, BrowserSession):
        raise ToolError("browsing is not available in this run")
    return session


def _check_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise ToolError("url must not be empty")
    scheme = cleaned.split(":", 1)[0].lower() if ":" in cleaned else ""
    if scheme not in ALLOWED_SCHEMES:
        raise ToolError(f"only http and https URLs are allowed, got {cleaned!r}")
    return cleaned


def _resolved_addresses(host: str, port: int) -> tuple[ipaddress._BaseAddress, ...]:
    """Every address a host resolves to, with any IPv6 zone suffix removed.

    A separate function so the tests can supply addresses without a network.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise ToolError(f"could not resolve {host!r}") from exc
    addresses: list[ipaddress._BaseAddress] = []
    for info in infos:
        raw = str(info[4][0]).split("%", 1)[0]
        try:
            addresses.append(ipaddress.ip_address(raw))
        except ValueError as exc:
            raise ToolError(f"{host!r} resolved to an unreadable address {raw!r}") from exc
    return tuple(dict.fromkeys(addresses))


def _is_non_public(address: ipaddress._BaseAddress) -> bool:
    """True for anything that is not a routable public destination."""
    return any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def _split_host(parsed, cleaned: str) -> tuple[str, int]:
    """Host and port, tolerating a bracketless IPv6 literal like ``http://::1/``.

    urllib treats that shape as a broken port, and refusing it as malformed would
    hide which rule it broke.  A model that writes it should get the same egress
    answer as ``http://[::1]/``.
    """
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    try:
        host, port = parsed.hostname, parsed.port
    except ValueError:
        host, port = None, None
    if host:
        return host, port or default_port
    raw = parsed.netloc.rsplit("@", 1)[-1].strip("[]").split("%", 1)[0]
    try:
        ipaddress.ip_address(raw)
    except ValueError as exc:
        raise ToolError(f"url has no host: {cleaned!r}") from exc
    return raw, default_port


def _check_destination(url: str) -> str:
    """Refuse a URL that points at this machine or the local network."""
    cleaned = _check_url(url)
    parsed = urlparse(cleaned)
    host, port = _split_host(parsed, cleaned)
    for address in _resolved_addresses(host, port):
        if _is_non_public(address):
            raise ToolError(
                f"refusing to fetch {cleaned}: {host!r} resolves to {address}, which is "
                "not a public address. The browser tools do not reach this machine, "
                "the local network or link-local services."
            )
    return cleaned


def _get(session: BrowserSession, url: str, *, limit: int) -> httpx.Response:
    """GET a URL, following redirects one hop at a time through the egress check."""
    current = _check_destination(url)
    response: httpx.Response | None = None
    for _ in range(MAX_REDIRECTS + 1):
        try:
            response = session.http().get(current)
        except httpx.HTTPError as exc:
            raise ToolError(f"could not fetch {current}: {exc.__class__.__name__}") from exc
        if not response.is_redirect:
            break
        location = response.headers.get("location", "").strip()
        if not location:
            raise ToolError(
                f"{current} answered HTTP {response.status_code} without a Location header"
            )
        current = _check_destination(urljoin(str(response.url), location))
    else:
        raise ToolError(f"too many redirects starting at {url} (limit {MAX_REDIRECTS})")
    assert response is not None  # the loop either breaks or raises
    if response.status_code >= 400:
        raise ToolError(f"{current} answered HTTP {response.status_code}")
    if len(response.content) > limit:
        raise ToolError(
            f"{current} returned {len(response.content)} bytes; the limit is {limit}"
        )
    return response


class OpenUrlArgs(_Args):
    url: str = Field(description="http(s) URL to open.")
    max_chars: int = Field(default=2000, ge=200, le=MAX_TEXT_CHARS)


class PageTextArgs(_Args):
    max_chars: int = Field(default=MAX_TEXT_CHARS, ge=200, le=MAX_TEXT_CHARS)


class SearchWebArgs(_Args):
    query: str = Field(description="Words to search the web for.")
    max_results: int = Field(default=5, ge=1, le=10)


class DownloadFileArgs(_Args):
    url: str = Field(description="http(s) URL of the file to download.")
    path: str = Field(description="Absolute path to write inside an allowed folder.")


def open_url(context: ToolContext, arguments: OpenUrlArgs) -> ToolResult:
    session = _session(context)
    url = _check_url(arguments.url)
    response = _get(session, url, limit=MAX_PAGE_BYTES)
    title, text = readable_text(response.text, limit=MAX_TEXT_CHARS)
    session.url, session.title, session.text, session.fetched = (
        str(response.url),
        title,
        text,
        len(text),
    )
    if context.dry_run:
        return ToolResult(
            text=f"[dry run] would open {session.url}",
            detail={"url": session.url, "dry_run": True},
        )
    heading = f"{session.url}\n{title}".strip()
    return ToolResult(
        text=f"{heading}\n\n{text[: arguments.max_chars]}",
        detail={
            "url": session.url,
            "title": title,
            "chars": len(text),
            "status": response.status_code,
        },
    )


def get_page_text(context: ToolContext, arguments: PageTextArgs) -> ToolResult:
    session = _session(context)
    if not session.url:
        raise ToolError("no page is open; call open_url first")
    return ToolResult(
        text=session.text[: arguments.max_chars] or "(the page had no readable text)",
        detail={"url": session.url, "title": session.title, "chars": len(session.text)},
    )


_RESULT_LINK = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def result_url(href: str) -> str:
    """The real destination of a search result link.

    The endpoint returns its own redirect wrapped around the target
    (``//duckduckgo.com/l/?uddg=<url-encoded target>``). Handing that to the
    model would cost a turn per result and land it on a redirect page, so the
    target is unwrapped here.
    """
    cleaned = html_module.unescape(href.strip())
    if cleaned.startswith("//"):
        cleaned = f"https:{cleaned}"
    parsed = urlparse(cleaned)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return cleaned


def search_web(context: ToolContext, arguments: SearchWebArgs) -> ToolResult:
    """Search the web and return links. Best-effort: an endpoint can change."""
    session = _session(context)
    query = arguments.query.strip()
    if not query:
        raise ToolError("query must not be empty")

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    if context.dry_run:
        return ToolResult(text=f"[dry run] would search the web for {query!r}", detail={"dry_run": True})

    response = _get(session, url, limit=MAX_PAGE_BYTES)
    found = _RESULT_LINK.findall(response.text)
    if not found:
        raise ToolError(
            "the search returned no parseable results; the endpoint may have changed"
        )

    lines: list[str] = []
    for title, href in ((t, result_url(h)) for h, t in found):
        if not href.startswith(("http://", "https://")):
            continue
        shown = readable_text(title)[1] or title.strip()
        lines.append(f"{len(lines) + 1}. {shown}\n   {href}")
        if len(lines) >= arguments.max_results:
            break
    if not lines:
        raise ToolError("the search returned no usable links")
    return ToolResult(
        text="\n".join(lines),
        detail={"query": query, "results": len(found), "shown": len(lines)},
    )


def download_file(context: ToolContext, arguments: DownloadFileArgs) -> ToolResult:
    session = _session(context)
    url = _check_url(arguments.url)
    target = context.sandbox.resolve(arguments.path)
    if target.exists():
        raise ToolError(f"{target} already exists; choose another name")
    if not target.parent.exists():
        raise ToolError(f"{target.parent} does not exist")

    if context.dry_run:
        return ToolResult(
            text=f"[dry run] would download {url} to {target}",
            detail={"url": url, "path": str(target), "dry_run": True},
        )

    response = _get(session, url, limit=MAX_DOWNLOAD_BYTES)
    target.write_bytes(response.content)
    return ToolResult(
        text=f"downloaded {len(response.content)} bytes from {url} to {target}",
        detail={
            "url": str(response.url),
            "path": str(target),
            "bytes": len(response.content),
            "content_type": response.headers.get("content-type", "unknown"),
        },
    )


__all__ = [
    "BrowserSession",
    "DownloadFileArgs",
    "OpenUrlArgs",
    "PageTextArgs",
    "SearchWebArgs",
    "download_file",
    "get_page_text",
    "open_url",
    "readable_text",
    "search_web",
]
